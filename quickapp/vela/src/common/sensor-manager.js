/**
 * sensor-manager.js - IMU传感器管理器
 * 负责加速度计的生命周期管理、数据采集、环形缓冲区
 *
 * OpenVela Sensor API:
 *   import sensor from '@system.sensor'
 *   sensor.subscribeAccelerometer({ interval, callback, fail })
 *   sensor.unsubscribeAccelerometer()
 */

import sensor from '@system.sensor'
import { SAMPLING, POWER } from './constants'

/** 传感器状态枚举 */
const STATE = {
  IDLE: 'idle',
  STARTING: 'starting',
  RUNNING: 'running',
  PAUSING: 'pausing',
  PAUSED: 'paused',
  STOPPING: 'stopping'
}

class SensorManager {
  constructor() {
    /** 当前传感器状态 */
    this._state = STATE.IDLE
    /** 原始数据环形缓冲区 [timestamp, x, y, z] */
    this._buffer = []
    /** 缓冲区最大容量 */
    this._maxBuffer = SAMPLING.WINDOW_SIZE * 2
    /** 当前采样间隔(ms) */
    this._interval = SAMPLING.DEFAULT_INTERVAL
    /** 数据回调函数列表 */
    this._listeners = []
    /** 状态变化回调 */
    this._stateListeners = []
    /** 后台暂停定时器 */
    this._bgTimer = null
    /** 数据丢弃计数(异常检测) */
    this._dropCount = 0
    /** 上一次数据时间戳 */
    this._lastTimestamp = 0
    /** 连续异常计数 */
    this._consecutiveErrors = 0
    /** 最大连续异常次数(超过则停止) */
    this._maxConsecutiveErrors = 10
  }

  /** 获取当前状态 */
  get state() { return this._state }

  /** 是否正在运行 */
  get isRunning() { return this._state === STATE.RUNNING }

  /** 获取最近N条原始数据 */
  getData(count) {
    if (count === undefined) count = this._buffer.length
    const start = Math.max(0, this._buffer.length - count)
    return this._buffer.slice(start)
  }

  /** 获取最新的单条数据 { x, y, z } */
  getLatest() {
    if (this._buffer.length === 0) return null
    const last = this._buffer[this._buffer.length - 1]
    return { x: last[1], y: last[2], z: last[3] }
  }

  /** 注册数据监听器 */
  onData(callback) {
    if (typeof callback === 'function') {
      this._listeners.push(callback)
    }
    return () => {
      this._listeners = this._listeners.filter(cb => cb !== callback)
    }
  }

  /** 注册状态变化监听器 */
  onStateChange(callback) {
    if (typeof callback === 'function') {
      this._stateListeners.push(callback)
    }
    return () => {
      this._stateListeners = this._stateListeners.filter(cb => cb !== callback)
    }
  }

  /** 启动传感器采集 */
  start(interval) {
    if (this._state === STATE.RUNNING || this._state === STATE.STARTING) {
      console.log('[SensorManager] 已在运行或启动中, 跳过')
      return
    }

    this._setState(STATE.STARTING)
    this._interval = interval || this._interval
    this._buffer = []
    this._consecutiveErrors = 0

    console.log('[SensorManager] 启动加速度计, interval=' + this._interval + 'ms')

    sensor.subscribeAccelerometer({
      interval: this._getIntervalMode(),
      callback: (data) => {
        this._onData(data)
      },
      fail: (data, code) => {
        console.error('[SensorManager] 订阅失败 code=' + code + ' data=' + data)
        this._consecutiveErrors++
        if (this._consecutiveErrors >= this._maxConsecutiveErrors) {
          console.error('[SensorManager] 连续异常过多, 停止采集')
          this.stop()
        }
      }
    })

    this._setState(STATE.RUNNING)
  }

  /** 停止传感器采集 */
  stop() {
    if (this._state === STATE.IDLE || this._state === STATE.STOPPING) {
      return
    }

    this._setState(STATE.STOPPING)
    console.log('[SensorManager] 停止加速度计')

    try {
      sensor.unsubscribeAccelerometer()
    } catch (e) {
      console.error('[SensorManager] 取消订阅异常: ' + e)
    }

    this._clearBgTimer()
    this._setState(STATE.IDLE)
  }

  /** 暂停采集(后台) */
  pause() {
    if (this._state !== STATE.RUNNING) return

    this._setState(STATE.PAUSING)
    console.log('[SensorManager] 暂停传感器(后台)')

    try {
      sensor.unsubscribeAccelerometer()
    } catch (e) {
      console.error('[SensorManager] 暂停异常: ' + e)
    }

    this._setState(STATE.PAUSED)
  }

  /** 恢复采集(前台) */
  resume() {
    if (this._state !== STATE.PAUSED) return

    console.log('[SensorManager] 恢复传感器(前台)')
    this.start()
  }

  /** 动态调整采样频率 */
  setInterval(interval) {
    if (interval === this._interval) return
    this._interval = interval
    if (this._state === STATE.RUNNING) {
      // 重启传感器以应用新频率
      this.stop()
      this.start(interval)
    }
  }

  /** 切换到省电模式(低采样率) */
  enablePowerSave() {
    this.setInterval(SAMPLING.SLOW_INTERVAL)
  }

  /** 恢复正常采样率 */
  disablePowerSave() {
    this.setInterval(SAMPLING.DEFAULT_INTERVAL)
  }

  /** 低电量回调 - 降频 */
  onLowBattery(level) {
    if (level <= POWER.LOW_BATTERY_THRESHOLD) {
      console.log('[SensorManager] 电量低(' + level + '%), 切换省电模式')
      this.enablePowerSave()
    }
  }

  /** 应用进入后台 - 延迟暂停 */
  onBackground() {
    this._clearBgTimer()
    this._bgTimer = setTimeout(() => {
      if (this._state === STATE.RUNNING) {
        this.pause()
      }
    }, POWER.BACKGROUND_PAUSE_DELAY)
  }

  /** 应用回到前台 - 恢复 */
  onForeground() {
    this._clearBgTimer()
    if (this._state === STATE.PAUSED) {
      this.resume()
    }
  }

  /** 清除后台定时器 */
  _clearBgTimer() {
    if (this._bgTimer !== null) {
      clearTimeout(this._bgTimer)
      this._bgTimer = null
    }
  }

  /** 处理传感器原始数据 */
  _onData(data) {
    const now = Date.now()

    // 异常时间戳检测(跳过过快的数据)
    if (this._lastTimestamp > 0 && (now - this._lastTimestamp) < 5) {
      this._dropCount++
      return
    }

    // NaN检测
    if (isNaN(data.x) || isNaN(data.y) || isNaN(data.z)) {
      this._consecutiveErrors++
      return
    }

    this._consecutiveErrors = 0
    this._lastTimestamp = now

    // 写入环形缓冲区
    const entry = [now, data.x, data.y, data.z]
    this._buffer.push(entry)

    // 缓冲区满时移除最旧数据(保留最近两倍窗口)
    if (this._buffer.length > this._maxBuffer) {
      this._buffer = this._buffer.slice(this._buffer.length - this._maxBuffer)
    }

    // 通知所有监听器
    for (let i = 0; i < this._listeners.length; i++) {
      try {
        this._listeners[i](data)
      } catch (e) {
        console.error('[SensorManager] 监听器异常: ' + e)
      }
    }
  }

  /** 将采样间隔映射为传感器interval模式 */
  _getIntervalMode() {
    if (this._interval <= 30) return 'game'
    if (this._interval <= 60) return 'normal'
    return 'ui'
  }

  /** 设置状态并通知 */
  _setState(state) {
    if (this._state === state) return
    const old = this._state
    this._state = state
    for (let i = 0; i < this._stateListeners.length; i++) {
      try {
        this._stateListeners[i](state, old)
      } catch (e) {
        console.error('[SensorManager] 状态监听器异常: ' + e)
      }
    }
  }

  /** 销毁 - 释放所有资源 */
  destroy() {
    this.stop()
    this._listeners = []
    this._stateListeners = []
    this._buffer = []
  }
}

// 导出单例
export default new SensorManager()
