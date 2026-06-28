package app.floatdeck.sensor

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Device posture sensor listener.
 *
 * Outputs:
 * - rollX: left/right tilt.
 * - pitchY: front/back tilt.
 * - yawZ: azimuth-like heading from Android orientation.
 * - twistZ: accumulated local screen-plane twist from frame-to-frame rotation changes.
 * - flatness: 0 when upright-ish, 1 when the phone is lying flat face-up/face-down.
 */
class SensorHandler(
    private val context: Context,
) : SensorEventListener {
    companion object {
        /**
         * 把旋转矢量传感器返回的 values 截断到前 4 个元素。
         * 部分厂商返回长度为 5（含 heading），会导致
         * [SensorManager.getRotationMatrixFromVector] 抛 IllegalArgumentException。
         */
        internal fun safeRotationValues(values: FloatArray): FloatArray =
            if (values.size > 4) values.copyOf(4) else values
    }

    /** 左右倾斜值（roll），约 -1 ~ 1 */
    var rollX = 0f
        private set

    /** 前后倾斜值（pitch），约 -1 ~ 1 */
    var pitchY = 0f
        private set

    /** 水平旋转值（yaw / azimuth），弧度制，约 -PI ~ PI */
    var yawZ = 0f
        private set

    /** 累积的屏幕平面旋转量，适合平躺时检测手机绕屏幕法线旋转。 */
    var twistZ = 0f
        private set

    /** 设备是否接近平放：0=竖/斜持，1=屏幕朝上/朝下接近平放 */
    var flatness = 0f
        private set

    private val sensorManager by lazy {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }

    private var registered = false
    private val currentRotationMatrix = FloatArray(9)
    private val previousRotationMatrix = FloatArray(9)
    private val angleChange = FloatArray(3)
    private var hasPreviousRotationMatrix = false

    /** 注册传感器监听。优先游戏旋转矢量，因为它的相对 yaw 更适合壁纸视差。 */
    fun register() {
        if (registered) return

        val gameRotation = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        if (gameRotation != null) {
            sensorManager.registerListener(this, gameRotation, SensorManager.SENSOR_DELAY_UI)
            registered = true
            return
        }

        val rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rotationVector != null) {
            sensorManager.registerListener(this, rotationVector, SensorManager.SENSOR_DELAY_UI)
            registered = true
            return
        }

        // 最后回退：仅用加速度计估算倾斜（无陀螺仪的设备）
        val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_UI)
            registered = true
        }
    }

    /** 取消传感器监听。 */
    fun unregister() {
        if (!registered) return
        sensorManager.unregisterListener(this)
        registered = false
        hasPreviousRotationMatrix = false
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ROTATION_VECTOR, Sensor.TYPE_GAME_ROTATION_VECTOR -> {
                SensorManager.getRotationMatrixFromVector(
                    currentRotationMatrix,
                    safeRotationValues(event.values),
                )

                if (hasPreviousRotationMatrix) {
                    SensorManager.getAngleChange(
                        angleChange,
                        currentRotationMatrix,
                        previousRotationMatrix,
                    )
                    twistZ += angleChange[2]
                }
                currentRotationMatrix.copyInto(previousRotationMatrix)
                hasPreviousRotationMatrix = true

                val orientation = FloatArray(3)
                SensorManager.getOrientation(currentRotationMatrix, orientation)
                yawZ = orientation[0]
                rollX = orientation[2]
                pitchY = orientation[1]

                // Matrix index 8 describes how much the phone Z axis points along world Z.
                // Its absolute value is close to 1 when the phone is lying flat, including face-down.
                flatness = abs(currentRotationMatrix[8]).coerceIn(0f, 1f)
            }
            Sensor.TYPE_ACCELEROMETER -> {
                val g = event.values
                val norm = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
                if (norm > 0.1f) {
                    rollX = g[0] / norm
                    pitchY = g[1] / norm
                    flatness = abs(g[2] / norm).coerceIn(0f, 1f)
                }
            }
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int,
    ) {}
}
