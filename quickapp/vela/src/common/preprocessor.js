/**
 * preprocessor.js - 信号预处理模块
 * 负责传感器原始数据的清洗、滤波、特征提取
 *
 * 功能:
 *   1. 异常值检测与过滤
 *   2. IIR低通滤波(去噪)
 *   3. Z-score标准化
 *   4. 时域特征提取(均值/方差/最大/最小/过零率)
 *   5. 加速度幅值计算(用于跌倒检测)
 */

import { SAMPLING } from './constants'

class Preprocessor {
  constructor() {
    /** IIR低通滤波器系数(截止频率~20Hz, 采样率25Hz) */
    this._alpha = 0.2
    /** 滤波后的上一帧值 */
    this._prevFiltered = [0, 0, 0]
    /** Z-score标准化参数 { mean, std } (每轴) */
    this._normParams = {
      mean: [0, 0, 0],
      std: [1, 1, 1]
    }
    /** 是否已初始化标准化参数 */
    this._normReady = false
    /** 滑动窗口数据(用于统计标准化参数) */
    this._calibBuffer = []
    /** 校准所需样本数 */
    this._calibSize = 200
    /** 重力加速度估计(用于去除重力分量) */
    this._gravity = [0, 0, 9.8]
    /** 重力估计的IIR系数(低频) */
    this._gravityAlpha = 0.05
  }

  /**
   * 处理单帧数据, 返回清洗后的 [x, y, z]
   * @param {{x: number, y: number, z: number}} raw - 原始加速度数据
   * @returns {{x, y, z, magnitude}|null} 清洗后数据, 异常时返回null
   */
  process(raw) {
    if (!raw || isNaN(raw.x) || isNaN(raw.y) || isNaN(raw.z)) {
      return null
    }

    // 1. 异常值检测: 加速度幅值应在合理范围 [0, 30] m/s²
    const mag = Math.sqrt(raw.x * raw.x + raw.y * raw.y + raw.z * raw.z)
    if (mag > 30 || mag < 0.1) {
      return null
    }

    // 2. IIR低通滤波(去高频噪声)
    const filtered = this._lowPassFilter(raw)

    // 3. 重力分量估计与去除(得到线性加速度)
    this._updateGravity(filtered)
    const linear = [
      filtered[0] - this._gravity[0],
      filtered[1] - this._gravity[1],
      filtered[2] - this._gravity[2]
    ]

    // 4. 校准阶段: 收集数据用于Z-score参数估计
    if (!this._normReady) {
      this._calibBuffer.push([linear[0], linear[1], linear[2]])
      if (this._calibBuffer.length >= this._calibSize) {
        this._computeNormParams()
      }
      // 校准期间返回原始滤波值(不标准化)
      return {
        x: linear[0],
        y: linear[1],
        z: linear[2],
        magnitude: mag
      }
    }

    // 5. Z-score标准化
    const norm = this._normalize(linear)

    return {
      x: norm[0],
      y: norm[1],
      z: norm[2],
      magnitude: mag
    }
  }

  /**
   * 从滑动窗口数据提取特征向量
   * @param {Array} window - 数据窗口 [[timestamp, x, y, z], ...]
   * @returns {Float32Array} 特征向量 [mean_x, var_x, max_x, min_x, zcr_x, ...]
   */
  extractFeatures(window) {
    const features = new Float32Array(SAMPLING.INPUT_DIMS * SAMPLING.FEATURE_DIMS)

    for (let axis = 0; axis < SAMPLING.INPUT_DIMS; axis++) {
      const col = new Float32Array(window.length)

      // 提取单轴数据
      for (let i = 0; i < window.length; i++) {
        col[i] = window[i][axis + 1] // +1 跳过timestamp
      }

      const base = axis * SAMPLING.FEATURE_DIMS

      // 均值
      features[base] = this._mean(col)
      // 方差
      features[base + 1] = this._variance(col, features[base])
      // 最大值
      features[base + 2] = this._max(col)
      // 最小值
      features[base + 3] = this._min(col)
      // 过零率(相对于均值)
      features[base + 4] = this._zeroCrossingRate(col, features[base])
    }

    return features
  }

  /**
   * 计算加速度幅值(用于跌倒检测辅助判断)
   * @param {Array} window - 数据窗口
   * @returns {{maxMag, avgMag, impactCount}} 幅值统计
   */
  getMagnitudeStats(window) {
    let maxMag = 0
    let sumMag = 0
    let impactCount = 0
    const impactThreshold = 20 // m/s²

    for (let i = 0; i < window.length; i++) {
      const x = window[i][1]
      const y = window[i][2]
      const z = window[i][3]
      const mag = Math.sqrt(x * x + y * y + z * z)

      if (mag > maxMag) maxMag = mag
      sumMag += mag
      if (mag > impactThreshold) impactCount++
    }

    return {
      maxMag: maxMag,
      avgMag: sumMag / window.length,
      impactCount: impactCount
    }
  }

  /**
   * IIR低通滤波
   * @param {Array} raw - [x, y, z]
   * @returns {Array} 滤波后的 [x, y, z]
   */
  _lowPassFilter(raw) {
    const result = [0, 0, 0]
    result[0] = this._alpha * raw.x + (1 - this._alpha) * this._prevFiltered[0]
    result[1] = this._alpha * raw.y + (1 - this._alpha) * this._prevFiltered[1]
    result[2] = this._alpha * raw.z + (1 - this._alpha) * this._prevFiltered[2]
    this._prevFiltered = result.slice()
    return result
  }

  /**
   * 重力分量估计(极低通滤波)
   * @param {Array} filtered - 滤波后的 [x, y, z]
   */
  _updateGravity(filtered) {
    for (let i = 0; i < 3; i++) {
      this._gravity[i] = this._gravityAlpha * filtered[i] + (1 - this._gravityAlpha) * this._gravity[i]
    }
  }

  /**
   * Z-score标准化
   * @param {Array} data - [x, y, z]
   * @returns {Array} 标准化后的 [x, y, z]
   */
  _normalize(data) {
    const result = [0, 0, 0]
    for (let i = 0; i < 3; i++) {
      const std = this._normParams.std[i]
      result[i] = std > 0.001 ? (data[i] - this._normParams.mean[i]) / std : 0
    }
    return result
  }

  /**
   * 从校准缓冲区计算Z-score参数
   */
  _computeNormParams() {
    const n = this._calibBuffer.length
    if (n === 0) return

    for (let axis = 0; axis < 3; axis++) {
      // 计算均值
      let sum = 0
      for (let i = 0; i < n; i++) {
        sum += this._calibBuffer[i][axis]
      }
      const mean = sum / n
      this._normParams.mean[axis] = mean

      // 计算标准差
      let sumSq = 0
      for (let i = 0; i < n; i++) {
        const diff = this._calibBuffer[i][axis] - mean
        sumSq += diff * diff
      }
      const std = Math.sqrt(sumSq / n)
      this._normParams.std[axis] = std > 0.001 ? std : 1
    }

    this._normReady = true
    this._calibBuffer = [] // 释放校准缓冲区
    console.log('[Preprocessor] 校准完成, mean=' + JSON.stringify(this._normParams.mean) +
      ' std=' + JSON.stringify(this._normParams.std))
  }

  // ========== 统计工具函数 ==========

  /** 均值 */
  _mean(arr) {
    let sum = 0
    for (let i = 0; i < arr.length; i++) sum += arr[i]
    return sum / arr.length
  }

  /** 方差 */
  _variance(arr, mean) {
    let sum = 0
    for (let i = 0; i < arr.length; i++) {
      const d = arr[i] - mean
      sum += d * d
    }
    return sum / arr.length
  }

  /** 最大值 */
  _max(arr) {
    let m = arr[0]
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > m) m = arr[i]
    }
    return m
  }

  /** 最小值 */
  _min(arr) {
    let m = arr[0]
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] < m) m = arr[i]
    }
    return m
  }

  /** 过零率(相对均值) */
  _zeroCrossingRate(arr, mean) {
    let count = 0
    for (let i = 1; i < arr.length; i++) {
      if ((arr[i] - mean) * (arr[i - 1] - mean) < 0) count++
    }
    return count / (arr.length - 1)
  }

  /** 重置(重新校准) */
  reset() {
    this._normReady = false
    this._calibBuffer = []
    this._gravity = [0, 0, 9.8]
    this._prevFiltered = [0, 0, 0]
    console.log('[Preprocessor] 已重置')
  }
}

// 导出单例
export default new Preprocessor()
