/**
 * constants.js - 全局常量定义
 * 姿态守护应用的所有配置常量集中管理
 */

/** 姿态标签枚举 */
export const POSTURE = {
  STANDING: 'standing',
  SITTING: 'sitting',
  WALKING: 'walking',
  HAND_RAISE: 'hand_raise',
  FALL: 'fall',
  UNKNOWN: 'unknown'
}

/** 姿态中文名映射 */
export const POSTURE_LABEL = {
  standing: '站立',
  sitting: '坐下',
  walking: '行走',
  hand_raise: '抬手',
  fall: '摔倒',
  unknown: '未知'
}

/** 姿态英文名映射 */
export const POSTURE_LABEL_EN = {
  standing: 'Standing',
  sitting: 'Sitting',
  walking: 'Walking',
  hand_raise: 'Hand Raise',
  fall: 'Fall',
  unknown: 'Unknown'
}

/** 姿态对应颜色 */
export const POSTURE_COLOR = {
  standing: '#4CAF50',
  sitting: '#2196F3',
  walking: '#FF9800',
  hand_raise: '#9C27B0',
  fall: '#F44336',
  unknown: '#757575'
}

/** 采样配置 */
export const SAMPLING = {
  /** 默认采样间隔(ms) */
  DEFAULT_INTERVAL: 40,
  /** 游戏模式采样间隔(ms) - 用于高精度场景 */
  FAST_INTERVAL: 20,
  /** 省电模式采样间隔(ms) */
  SLOW_INTERVAL: 100,
  /** 滑动窗口大小(样本数) */
  WINDOW_SIZE: 50,
  /** 传感器数据维度(加速度3轴) */
  INPUT_DIMS: 3,
  /** 特征维度: 均值+方差+最大值+最小值+过零率 = 5 */
  FEATURE_DIMS: 5
}

/** 模型配置 */
export const MODEL = {
  /** CNN输入特征维度 = WINDOW_SIZE * FEATURE_DIMS_PER_AXIS */
  /** 每轴5个特征(均值/方差/最大/最小/过零率)，3轴共15个特征 */
  INPUT_SIZE: 15,
  /** CNN卷积核大小 */
  CONV_KERNEL_SIZE: 5,
  /** 第一层卷积输出通道数 */
  CONV1_CHANNELS: 8,
  /** 第二层卷积输出通道数 */
  CONV2_CHANNELS: 16,
  /** 全连接隐藏层大小 */
  FC_HIDDEN: 24,
  /** 输出类别数 */
  NUM_CLASSES: 5,
  /** 推理置信度阈值 - 低于此值视为未知 */
  CONFIDENCE_THRESHOLD: 0.4,
  /** 投票平滑窗口大小 */
  SMOOTH_WINDOW: 5
}

/** 功耗管理配置 */
export const POWER = {
  /** 低电量阈值(%) */
  LOW_BATTERY_THRESHOLD: 20,
  /** 低电量时采样间隔(ms) */
  LOW_BATTERY_INTERVAL: 200,
  /** 应用后台后传感器暂停延迟(ms) */
  BACKGROUND_PAUSE_DELAY: 5000,
  /** 空闲超时暂停传感器(ms) - 无姿态变化时 */
  IDLE_TIMEOUT: 30000
}

/** 存储键名 */
export const STORAGE_KEY = {
  /** 历史记录 */
  HISTORY: 'posture_history',
  /** 设置 */
  SETTINGS: 'posture_settings',
  /** 模型权重缓存 */
  MODEL_CACHE: 'posture_model'
}

/** 振动模式 */
export const VIBRATE = {
  /** 摔倒警告振动 - 长震动 */
  FALL_ALERT_MODE: 'long',
  /** 摔倒警告振动次数 */
  FALL_ALERT_COUNT: 3,
  /** 摔倒警告振动时长(ms) */
  FALL_ALERT_DURATION: 500,
  /** 摔倒警告振动间隔(ms) */
  FALL_ALERT_INTERVAL: 300
}
