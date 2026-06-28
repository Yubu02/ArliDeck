package app.floatdeck.gl

import android.content.Context
import android.graphics.BitmapFactory
import android.graphics.Color
import android.opengl.GLES30
import android.opengl.Matrix
import app.floatdeck.data.TemplateConfig
import app.floatdeck.data.TemplateLayoutConfig
import java.nio.FloatBuffer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin

// ============================================================================
// 数据模型
// ============================================================================

/** 单张立绘卡片的运行时状态。 */
data class PortraitState(
    val id: String,
    val label: String,
    var textureId: Int = 0,
    var textureWidth: Int = 0,
    var textureHeight: Int = 0,
    var drawOrder: Int = 0,
    var offsetX: Float = 0f,
    var offsetY: Float = 0f,
    var velocityX: Float = 0f,
    var velocityY: Float = 0f,
)

// ============================================================================
// 锁屏布局配置
// ============================================================================

/** 锁屏时的行排列定义。 */
data class LockLayout(
    val rows: List<Int>,
)

private val LANDSCAPE_LAYOUTS =
    listOf(
        LockLayout(listOf(4, 4, 4)),
        LockLayout(listOf(5, 2, 5)),
    )

private val PORTRAIT_LAYOUTS =
    listOf(
        LockLayout(listOf(4, 4, 4)),
        LockLayout(listOf(3, 3, 3, 3)),
    )

// ============================================================================
// 主渲染器
// ============================================================================

/** FloatDeck 动态壁纸的 OpenGL ES 3.0 渲染器。 */
class FloatDeckRenderer(
    private val context: Context,
) {
    private var portraitProgram = 0
    private var backgroundProgram = 0
    private var quadVertexBuffer: FloatBuffer? = null

    private var wallpaperTextureId = 0
    private var wallpaperPixelWidth = 1
    private var wallpaperPixelHeight = 1

    private val portraitStates = mutableListOf<PortraitState>()
    private var templateLayout = TemplateLayoutConfig()

    var smoothedRollX = 0f
    var smoothedPitchY = 0f
    var smoothedYawZ = 0f
    var smoothedTwistZ = 0f
    var smoothedFlatness = 0f

    private var yawCenter = 0f
    private var twistCenter = 0f
    private var axisCenter = 0f
    private var needsOrientationRecenter = true

    var transitionProgress = 0f
    var targetTransition = 0f
    private var isFirstFrame = true

    private var swayTimeSeconds = 0f

    private var currentLockLayout: LockLayout = PORTRAIT_LAYOUTS[0]
    private var layoutSelected = false
    private var needsTemplateReload = false

    private var screenWidthPixels = 1f
    private var screenHeightPixels = 1f
    private val orthographicMatrix = FloatArray(16)

    private var uniformPortraitMvp = 0
    private var uniformPortraitOffset = 0
    private var uniformPortraitRotation = 0
    private var uniformPortraitScale = 0
    private var uniformPortraitParallax = 0
    private var uniformPortraitTexture = 0
    private var uniformPortraitAlpha = 0
    private var uniformPortraitShadowColor = 0
    private var uniformPortraitShadowOffset = 0
    private var uniformPortraitCornerRadius = 0
    private var uniformPortraitEffect = 0
    private var uniformPortraitTime = 0
    private var uniformPortraitViewAngle = 0

    private var uniformBackgroundMvp = 0
    private var uniformBackgroundParallax = 0
    private var uniformBackgroundTexture = 0
    private var uniformBackgroundAlpha = 0

    private val cardPortraitHeightScreenFraction = 0.24f
    private val cardPortraitAspectRatio = 0.5f

    /** 当前立绘特效。由壁纸服务从 SharedPreferences 读取后设置。 */
    var portraitEffect: Int = 0

    var draggedPortraitIndex = -1
    var previousTouchX = 0f
    var previousTouchY = 0f

    private val unlockDelaySeconds = 0.1f
    private var unlockDelayTimer = 0f
    private var isWaitingForUnlock = false

    private val placeholderColors =
        intArrayOf(
            Color.argb(255, 70, 50, 90),
            Color.argb(255, 50, 60, 90),
            Color.argb(255, 60, 45, 85),
            Color.argb(255, 55, 55, 100),
            Color.argb(255, 75, 40, 80),
            Color.argb(255, 45, 65, 95),
            Color.argb(255, 65, 50, 75),
            Color.argb(255, 50, 70, 85),
            Color.argb(255, 80, 45, 70),
            Color.argb(255, 55, 55, 90),
            Color.argb(255, 60, 60, 80),
            Color.argb(255, 45, 50, 100),
        )

    fun onSurfaceCreated(
        gl: javax.microedition.khronos.opengles.GL10?,
        config: javax.microedition.khronos.egl.EGLConfig?,
    ) {
        GLES30.glClearColor(0.1f, 0.1f, 0.15f, 1f)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        portraitProgram = ShaderProgram.compile(Shaders.portraitVertex, Shaders.portraitFragment)
        backgroundProgram = ShaderProgram.compile(Shaders.backgroundVertex, Shaders.backgroundFragment)

        uniformPortraitMvp = GLES30.glGetUniformLocation(portraitProgram, "uMVP")
        uniformPortraitOffset = GLES30.glGetUniformLocation(portraitProgram, "uOffset")
        uniformPortraitRotation = GLES30.glGetUniformLocation(portraitProgram, "uRotation")
        uniformPortraitScale = GLES30.glGetUniformLocation(portraitProgram, "uScale")
        uniformPortraitParallax = GLES30.glGetUniformLocation(portraitProgram, "uParallax")
        uniformPortraitTexture = GLES30.glGetUniformLocation(portraitProgram, "uTexture")
        uniformPortraitAlpha = GLES30.glGetUniformLocation(portraitProgram, "uAlpha")
        uniformPortraitShadowColor = GLES30.glGetUniformLocation(portraitProgram, "uShadowColor")
        uniformPortraitShadowOffset = GLES30.glGetUniformLocation(portraitProgram, "uShadowOffset")
        uniformPortraitCornerRadius = GLES30.glGetUniformLocation(portraitProgram, "uRadius")
        uniformPortraitEffect = GLES30.glGetUniformLocation(portraitProgram, "uEffect")
        uniformPortraitTime = GLES30.glGetUniformLocation(portraitProgram, "uTime")
        uniformPortraitViewAngle = GLES30.glGetUniformLocation(portraitProgram, "uViewAngle")

        uniformBackgroundMvp = GLES30.glGetUniformLocation(backgroundProgram, "uMVP")
        uniformBackgroundParallax = GLES30.glGetUniformLocation(backgroundProgram, "uParallax")
        uniformBackgroundTexture = GLES30.glGetUniformLocation(backgroundProgram, "uTexture")
        uniformBackgroundAlpha = GLES30.glGetUniformLocation(backgroundProgram, "uAlpha")

        quadVertexBuffer = Quad.createBuffer()
    }

    fun onSurfaceChanged(
        gl: javax.microedition.khronos.opengles.GL10?,
        width: Int,
        height: Int,
    ) {
        screenWidthPixels = width.toFloat()
        screenHeightPixels = height.toFloat()
        GLES30.glViewport(0, 0, width, height)
        Matrix.orthoM(orthographicMatrix, 0, 0f, screenWidthPixels, screenHeightPixels, 0f, -1f, 1f)
        needsTemplateReload = true
    }

    private fun selectRandomLayout() {
        val isLandscape = screenWidthPixels > screenHeightPixels
        val layouts = if (isLandscape) LANDSCAPE_LAYOUTS else PORTRAIT_LAYOUTS
        currentLockLayout = layouts[kotlin.random.Random.nextInt(layouts.size)]
        layoutSelected = true
    }

    fun loadTemplate(template: TemplateConfig) {
        portraitStates.forEach { TextureLoader.deleteTexture(it.textureId) }
        portraitStates.clear()
        if (wallpaperTextureId != 0) TextureLoader.deleteTexture(wallpaperTextureId)
        templateLayout = template.layout
        requestOrientationRecenter()

        val bgResult =
            if (template.isRemote && template.wallpaperAsset != null) {
                loadTextureFromPath(template.wallpaperAsset)
            } else {
                loadFullResTexture(template.wallpaperAsset ?: "")
            }
        if (bgResult != null) {
            wallpaperTextureId = bgResult.first
            wallpaperPixelWidth = bgResult.second
            wallpaperPixelHeight = bgResult.third
        } else {
            wallpaperTextureId = TextureLoader.createGradientBackground()
            wallpaperPixelWidth = 512
            wallpaperPixelHeight = 1024
        }

        template.portraits.forEachIndexed { index, config ->
            val result =
                if (config.isRemote) {
                    loadTextureFromPath(config.assetPath)
                } else {
                    loadFullResTexture(config.assetPath)
                }
            if (result != null) {
                portraitStates.add(
                    PortraitState(
                        id = config.id,
                        label = config.label,
                        textureId = result.first,
                        textureWidth = result.second,
                        textureHeight = result.third,
                        drawOrder = index,
                    ),
                )
            } else {
                portraitStates.add(
                    PortraitState(
                        id = config.id,
                        label = config.label,
                        textureId =
                            TextureLoader.createPlaceholderTexture(
                                config.label,
                                bgColor = placeholderColors[index % placeholderColors.size],
                            ),
                        textureWidth = 128,
                        textureHeight = 256,
                        drawOrder = index,
                    ),
                )
            }
        }
    }

    private fun loadTextureFromPath(path: String): Triple<Int, Int, Int>? {
        return try {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(path, opts)
            val width = opts.outWidth
            val height = opts.outHeight
            val bitmap = BitmapFactory.decodeFile(path) ?: return null
            val texId = TextureLoader.loadBitmap(bitmap)
            bitmap.recycle()
            Triple(texId, width, height)
        } catch (_: Exception) {
            null
        }
    }

    private fun loadFullResTexture(assetPath: String): Triple<Int, Int, Int>? {
        return try {
            val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            context.assets.open(assetPath).use {
                BitmapFactory.decodeStream(it, null, opts)
            }
            val width = opts.outWidth
            val height = opts.outHeight
            if (width <= 0 || height <= 0) return null
            val bitmap =
                context.assets.open(assetPath).use {
                    BitmapFactory.decodeStream(it)
                } ?: return null
            val texId = TextureLoader.loadBitmap(bitmap)
            bitmap.recycle()
            Triple(texId, width, height)
        } catch (_: Exception) {
            null
        }
    }

    fun onDrawFrame(gl: javax.microedition.khronos.opengles.GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)

        if (isFirstFrame) {
            transitionProgress = targetTransition
            isFirstFrame = false
        }

        if (needsOrientationRecenter) {
            recenterOrientation()
        }

        if (isWaitingForUnlock) {
            unlockDelayTimer += FRAME_TIME_SECONDS
            if (unlockDelayTimer >= unlockDelaySeconds) {
                isWaitingForUnlock = false
                targetTransition = 0f
            }
        }

        val diff = targetTransition - transitionProgress
        transitionProgress += if (abs(diff) < 0.01f) diff else diff * 0.08f

        swayTimeSeconds += FRAME_TIME_SECONDS
        updateInertia()

        if (needsTemplateReload) {
            needsTemplateReload = false
            selectRandomLayout()
        }

        drawBackgroundLayers()
        drawPortraits()
    }

    private fun updateInertia() {
        val maxOffsetX = screenWidthPixels * 0.4f
        val maxOffsetY = screenHeightPixels * 0.4f

        portraitStates.forEach { state ->
            if (state.velocityX != 0f || state.velocityY != 0f) {
                state.offsetX += state.velocityX
                state.offsetY += state.velocityY
                state.velocityX *= 0.92f
                state.velocityY *= 0.92f
                if (abs(state.velocityX) < 0.5f) state.velocityX = 0f
                if (abs(state.velocityY) < 0.5f) state.velocityY = 0f
            }
            state.offsetX = state.offsetX.coerceIn(-maxOffsetX, maxOffsetX)
            state.offsetY = state.offsetY.coerceIn(-maxOffsetY, maxOffsetY)
        }
    }

    private fun drawBackgroundLayers() {
        val flatBlend = hybridFlatBlend()
        val flatX = flatTwistForParallax()
        val clampedRoll = lerp(
            smoothedRollX.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT),
            flatX,
            flatBlend,
        )
        val clampedPitch = smoothedPitchY.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT)
        val parallaxX = clampedRoll * screenWidthPixels * templateLayout.backgroundParallaxX
        val parallaxY = clampedPitch * screenHeightPixels * templateLayout.backgroundParallaxY
        drawSingleBackgroundLayer(parallaxX, parallaxY, 1.0f)
    }

    private fun drawSingleBackgroundLayer(
        parallaxX: Float,
        parallaxY: Float,
        alpha: Float,
    ) {
        GLES30.glUseProgram(backgroundProgram)
        val buffer = quadVertexBuffer ?: return
        bindQuadAttributes(buffer)

        val maxParallaxOffset = maxOf(abs(parallaxX), abs(parallaxY))
        val overscan = 1f + (maxParallaxOffset / minOf(screenWidthPixels, screenHeightPixels)) * 2f

        val screenAspect = screenWidthPixels / screenHeightPixels
        val textureAspect = wallpaperPixelWidth.toFloat() / wallpaperPixelHeight.toFloat()
        val scaleFactor =
            if (screenAspect > textureAspect) {
                screenWidthPixels * overscan / wallpaperPixelWidth.toFloat()
            } else {
                screenHeightPixels * overscan / wallpaperPixelHeight.toFloat()
            }
        val drawWidth = wallpaperPixelWidth.toFloat() * scaleFactor
        val drawHeight = wallpaperPixelHeight.toFloat() * scaleFactor

        val mvp =
            buildModelMatrix(
                centerX = screenWidthPixels / 2f + parallaxX,
                centerY = screenHeightPixels / 2f + parallaxY,
                width = drawWidth,
                height = drawHeight,
                rotation = 0f,
            )

        GLES30.glUniformMatrix4fv(uniformBackgroundMvp, 1, false, mvp, 0)
        GLES30.glUniform2f(uniformBackgroundParallax, 0f, 0f)
        GLES30.glUniform1f(uniformBackgroundAlpha, alpha)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, wallpaperTextureId)
        GLES30.glUniform1i(uniformBackgroundTexture, 0)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        unbindQuadAttributes()
    }

    private fun drawPortraits() {
        if (templateLayout.isFullscreen) {
            drawFullscreenPortraits()
        } else {
            drawCardPortraits()
        }
    }

    private fun drawFullscreenPortraits() {
        if (portraitStates.isEmpty()) return

        val centerRatioX = lerp(templateLayout.centerX, templateLayout.lockedCenterX, transitionProgress)
        val centerRatioY = lerp(templateLayout.centerY, templateLayout.lockedCenterY, transitionProgress)
        val heightRatio = lerp(templateLayout.portraitHeight, templateLayout.lockedPortraitHeight, transitionProgress)
        val rotation = lerp(templateLayout.rotation, templateLayout.lockedRotation, transitionProgress)

        val flatBlend = hybridFlatBlend()
        val flatX = flatTwistForParallax()
        val parallaxInputX = lerp(
            smoothedRollX.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT),
            flatX,
            flatBlend,
        )
        val parallaxX = parallaxInputX * screenWidthPixels * templateLayout.portraitParallaxX
        val parallaxY = smoothedPitchY.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT) *
            screenHeightPixels * templateLayout.portraitParallaxY
        val centerX = centerRatioX * screenWidthPixels + parallaxX
        val centerY = centerRatioY * screenHeightPixels + parallaxY
        val height = screenHeightPixels * heightRatio

        val alphaPair = fullscreenAlphaPair()
        portraitStates.forEachIndexed { index, state ->
            val alpha =
                when (index) {
                    0 -> alphaPair.first
                    1 -> alphaPair.second
                    else -> 1f
                }
            drawPortraitQuad(
                state = state,
                centerX = centerX + state.offsetX,
                centerY = centerY + state.offsetY,
                height = height,
                rotation = rotation,
                alpha = alpha,
                rounded = false,
                shadow = false,
            )
        }
    }

    private fun fullscreenAlphaPair(): Pair<Float, Float> {
        if (portraitStates.size < 2) return Pair(1f, 0f)

        val normalizedTilt =
            if (templateLayout.tiltMode.equals("hybrid", ignoreCase = true)) {
                hybridNormalizedTilt()
            } else {
                axisNormalizedTilt()
            }
        val finalTilt = if (templateLayout.invertTilt) -normalizedTilt else normalizedTilt
        val rightAlpha = ((finalTilt + 1f) / 2f).coerceIn(0f, 1f)
        return Pair(1f - rightAlpha, rightAlpha)
    }

    private fun axisNormalizedTilt(): Float {
        val axisValue = currentAxisValue()
        val range = templateLayout.crossfadeRange.coerceAtLeast(0.01f)
        return (axisValue / range).coerceIn(-1f, 1f)
    }

    private fun hybridNormalizedTilt(): Float {
        val axisRange = templateLayout.crossfadeRange.coerceAtLeast(0.01f)
        val yawRange = templateLayout.yawRange.coerceAtLeast(0.01f)
        val axisDelta = currentAxisValue() - axisCenter
        val twistDelta = localTwistDelta()
        val axisTilt = (axisDelta / axisRange).coerceIn(-1f, 1f)
        val twistTilt = (twistDelta / yawRange).coerceIn(-1f, 1f)
        return lerp(axisTilt, twistTilt, hybridFlatBlend()).coerceIn(-1f, 1f)
    }

    private fun currentAxisValue(): Float =
        if (templateLayout.tiltAxis.equals("pitch", ignoreCase = true)) {
            smoothedPitchY
        } else {
            smoothedRollX
        }

    private fun hybridFlatBlend(): Float {
        if (!templateLayout.tiltMode.equals("hybrid", ignoreCase = true)) return 0f
        return smoothStep(
            templateLayout.flatnessStart,
            templateLayout.flatnessEnd,
            smoothedFlatness,
        )
    }

    private fun localTwistDelta(): Float {
        val twistDelta = smoothedTwistZ - twistCenter
        return if (abs(twistDelta) > 0.0005f) {
            twistDelta
        } else {
            angleDelta(smoothedYawZ, yawCenter)
        }
    }

    private fun flatTwistForParallax(): Float {
        val yawRange = templateLayout.yawRange.coerceAtLeast(0.01f)
        return ((localTwistDelta() / yawRange) * MAX_SENSOR_SHIFT).coerceIn(
            -MAX_SENSOR_SHIFT,
            MAX_SENSOR_SHIFT,
        )
    }

    fun requestOrientationRecenter() {
        needsOrientationRecenter = true
    }

    private fun recenterOrientation() {
        axisCenter = currentAxisValue()
        yawCenter = smoothedYawZ
        twistCenter = smoothedTwistZ
        needsOrientationRecenter = false
    }

    private fun drawCardPortraits() {
        GLES30.glUseProgram(portraitProgram)
        val sortedByZ = portraitStates.sortedBy { it.drawOrder }
        val totalCount = portraitStates.size
        val leftSideCount = (totalCount + 1) / 2

        sortedByZ.forEach { state ->
            val globalIndex = portraitStates.indexOf(state)
            val isOnLeftSide = globalIndex < leftSideCount
            val sideIndex = if (isOnLeftSide) globalIndex else globalIndex - leftSideCount
            val sideCount = if (isOnLeftSide) leftSideCount else totalCount - leftSideCount

            val (lockedCenterX, lockedCenterY) = calculateLockedPosition(globalIndex, totalCount)
            val lockedScale = 0.85f
            val lockedRotation = (globalIndex - totalCount / 2) * 2.5f

            val unlockedEdgeX = if (isOnLeftSide) 0.05f else 0.95f
            val unlockedEdgeY = calculateUnlockedY(sideIndex, sideCount)
            val unlockedScale = 0.64f
            val unlockedRotation =
                if (isOnLeftSide) -7.2f + sideIndex * 1.8f else 7.2f - sideIndex * 1.8f

            var drawX = lerp(unlockedEdgeX, lockedCenterX, transitionProgress) * screenWidthPixels
            var drawY = lerp(unlockedEdgeY, lockedCenterY, transitionProgress) * screenHeightPixels
            val drawScale = lerp(unlockedScale, lockedScale, transitionProgress)
            val drawRotation = lerp(unlockedRotation, lockedRotation, transitionProgress)

            drawX += smoothedPitchY.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT) * screenWidthPixels * 0.25f
            drawY += smoothedRollX.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT) * screenHeightPixels * 0.2f
            drawX += calculateSway(globalIndex)[0]
            drawY += calculateSway(globalIndex)[1]
            drawX += state.offsetX
            drawY += state.offsetY

            val portraitHeightPixels = screenHeightPixels * cardPortraitHeightScreenFraction * drawScale
            drawPortraitQuad(
                state = state,
                centerX = drawX,
                centerY = drawY,
                height = portraitHeightPixels,
                rotation = drawRotation,
                alpha = 1f,
                rounded = true,
                shadow = true,
                forcedAspectRatio = cardPortraitAspectRatio,
            )
        }
    }

    private fun drawPortraitQuad(
        state: PortraitState,
        centerX: Float,
        centerY: Float,
        height: Float,
        rotation: Float,
        alpha: Float,
        rounded: Boolean,
        shadow: Boolean,
        forcedAspectRatio: Float? = null,
    ) {
        GLES30.glUseProgram(portraitProgram)
        val buffer = quadVertexBuffer ?: return
        val aspectRatio = forcedAspectRatio ?: state.safeAspectRatio()
        val width = height * aspectRatio
        val mvp = buildModelMatrix(centerX, centerY, width, height, rotation)

        bindQuadAttributes(buffer)
        GLES30.glUniformMatrix4fv(uniformPortraitMvp, 1, false, mvp, 0)
        GLES30.glUniform2f(uniformPortraitOffset, 0f, 0f)
        GLES30.glUniform1f(uniformPortraitRotation, 0f)
        GLES30.glUniform2f(uniformPortraitScale, 1f, 1f)
        GLES30.glUniform2f(uniformPortraitParallax, 0f, 0f)
        GLES30.glUniform1f(uniformPortraitAlpha, alpha.coerceIn(0f, 1f))
        GLES30.glUniform4f(uniformPortraitShadowColor, 0f, 0f, 0f, if (shadow) 0.15f else 0f)
        GLES30.glUniform2f(uniformPortraitShadowOffset, 0.01f, -0.01f)
        GLES30.glUniform1f(uniformPortraitCornerRadius, if (rounded) 0.15f else 0f)
        GLES30.glUniform1i(uniformPortraitEffect, portraitEffect)
        GLES30.glUniform1f(uniformPortraitTime, swayTimeSeconds)
        GLES30.glUniform2f(
            uniformPortraitViewAngle,
            smoothedPitchY.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT),
            smoothedRollX.coerceIn(-MAX_SENSOR_SHIFT, MAX_SENSOR_SHIFT),
        )
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, state.textureId)
        GLES30.glUniform1i(uniformPortraitTexture, 0)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        unbindQuadAttributes()
    }

    private fun buildModelMatrix(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        rotation: Float,
    ): FloatArray {
        val model = FloatArray(16)
        val mvp = FloatArray(16)
        Matrix.setIdentityM(model, 0)
        Matrix.translateM(model, 0, centerX, centerY, 0f)
        Matrix.rotateM(model, 0, rotation, 0f, 0f, 1f)
        Matrix.scaleM(model, 0, width / 2f, height / 2f, 1f)
        Matrix.multiplyMM(mvp, 0, orthographicMatrix, 0, model, 0)
        return mvp
    }

    private fun calculateLockedPosition(
        index: Int,
        totalCount: Int,
    ): Pair<Float, Float> {
        if (!layoutSelected && portraitStates.isNotEmpty()) selectRandomLayout()

        val rows = currentLockLayout.rows
        var remaining = index
        var targetRow = 0
        var colInRow = 0
        for ((rowIdx, count) in rows.withIndex()) {
            if (remaining < count) {
                targetRow = rowIdx
                colInRow = remaining
                break
            }
            remaining -= count
        }

        val totalRows = rows.size
        val colsInRow = rows[targetRow]
        val rowWidth = (colsInRow - 1) * 0.1f
        val startX = 0.5f - rowWidth / 2f
        val xRatio = (startX + colInRow * 0.1f).coerceIn(0.05f, 0.95f)
        val yRatio =
            (0.2f + targetRow * (0.6f / (totalRows - 1).coerceAtLeast(1))).coerceIn(0.1f, 0.9f)
        return Pair(xRatio, yRatio)
    }

    private fun calculateUnlockedY(
        sideIndex: Int,
        sideCount: Int,
    ): Float {
        val yStart = 0.12f
        val yEnd = 0.88f
        val yStep = (yEnd - yStart) / (sideCount - 1).coerceAtLeast(1)
        return (yStart + sideIndex * yStep).coerceIn(0.1f, 0.9f)
    }

    private fun calculateSway(portraitIndex: Int): FloatArray {
        val seed = portraitIndex * 2654435761L
        val amplitudeVariation = ((seed and 0xFF) % 40 - 20) / 100f
        val frequencyVariation = ((seed shr 8 and 0xFF) % 30 - 15) / 100f
        val phaseOffset = (seed shr 16 and 0xFF) / 255f * 3.14f
        val portraitHeight = screenHeightPixels * cardPortraitHeightScreenFraction * 0.85f
        val amplitude = portraitHeight * 0.08f * (1f + amplitudeVariation)
        val frequency = 0.8f * (1f + frequencyVariation)
        val offsetY = sin(swayTimeSeconds * frequency + phaseOffset) * amplitude
        return floatArrayOf(0f, offsetY)
    }

    private fun bindQuadAttributes(buffer: FloatBuffer) {
        buffer.position(0)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, Quad.getStride(), buffer)
        buffer.position(Quad.getUvOffset() / 4)
        GLES30.glVertexAttribPointer(1, 2, GLES30.GL_FLOAT, false, Quad.getStride(), buffer)
    }

    private fun unbindQuadAttributes() {
        GLES30.glDisableVertexAttribArray(0)
        GLES30.glDisableVertexAttribArray(1)
    }

    fun release() {
        portraitStates.forEach { TextureLoader.deleteTexture(it.textureId) }
        portraitStates.clear()
        if (wallpaperTextureId != 0) TextureLoader.deleteTexture(wallpaperTextureId)
        if (portraitProgram != 0) GLES30.glDeleteProgram(portraitProgram)
        if (backgroundProgram != 0) GLES30.glDeleteProgram(backgroundProgram)
    }

    fun onTouchDown(
        touchX: Float,
        touchY: Float,
    ): Boolean {
        if (templateLayout.isFullscreen) return false

        val sortedByZ = portraitStates.sortedByDescending { it.drawOrder }
        for (state in sortedByZ) {
            val bounds = getPortraitBounds(state)
            if (touchX >= bounds[0] && touchX <= bounds[2] && touchY >= bounds[1] && touchY <= bounds[3]) {
                draggedPortraitIndex = portraitStates.indexOf(state)
                val maxOrder = portraitStates.maxOf { it.drawOrder }
                state.drawOrder = maxOrder + 1
                previousTouchX = touchX
                previousTouchY = touchY
                return true
            }
        }
        return false
    }

    fun onTouchMove(
        touchX: Float,
        touchY: Float,
    ) {
        if (draggedPortraitIndex < 0 || draggedPortraitIndex >= portraitStates.size) return
        val state = portraitStates[draggedPortraitIndex]
        val deltaX = touchX - previousTouchX
        val deltaY = touchY - previousTouchY
        state.offsetX += deltaX
        state.offsetY += deltaY
        state.velocityX = deltaX
        state.velocityY = deltaY
        previousTouchX = touchX
        previousTouchY = touchY
    }

    fun onTouchUp() {
        draggedPortraitIndex = -1
    }

    fun onDoubleTap(
        touchX: Float,
        touchY: Float,
    ): Boolean {
        if (templateLayout.isFullscreen) return false

        val sortedByZ = portraitStates.sortedByDescending { it.drawOrder }
        for (state in sortedByZ) {
            val bounds = getPortraitBounds(state)
            if (touchX >= bounds[0] && touchX <= bounds[2] && touchY >= bounds[1] && touchY <= bounds[3]) {
                state.offsetX = 0f
                state.offsetY = 0f
                state.velocityX = 0f
                state.velocityY = 0f
                return true
            }
        }
        return false
    }

    fun triggerUnlock() {
        isWaitingForUnlock = true
        unlockDelayTimer = 0f
        if (templateLayout.recenterOnScreenOn) {
            requestOrientationRecenter()
        }
    }

    fun triggerLock() {
        isWaitingForUnlock = false
        targetTransition = 1f
        selectRandomLayout()
        if (templateLayout.recenterOnScreenOn) {
            requestOrientationRecenter()
        }
    }

    private fun getPortraitBounds(state: PortraitState): FloatArray {
        val totalCount = portraitStates.size
        val leftSideCount = (totalCount + 1) / 2
        val globalIndex = portraitStates.indexOf(state)
        val isOnLeftSide = globalIndex < leftSideCount
        val sideIndex = if (isOnLeftSide) globalIndex else globalIndex - leftSideCount
        val sideCount = if (isOnLeftSide) leftSideCount else totalCount - leftSideCount
        val (lockedX, lockedY) = calculateLockedPosition(globalIndex, totalCount)
        val unlockedX = if (isOnLeftSide) 0.05f else 0.95f
        val unlockedY = calculateUnlockedY(sideIndex, sideCount)
        val centerX = lerp(unlockedX, lockedX, transitionProgress) * screenWidthPixels + state.offsetX
        val centerY = lerp(unlockedY, lockedY, transitionProgress) * screenHeightPixels + state.offsetY
        val scale = lerp(0.64f, 0.85f, transitionProgress)
        val height = screenHeightPixels * cardPortraitHeightScreenFraction * scale
        val width = height * cardPortraitAspectRatio

        return floatArrayOf(
            centerX - width / 2,
            centerY - height / 2,
            centerX + width / 2,
            centerY + height / 2,
        )
    }

    private fun PortraitState.safeAspectRatio(): Float {
        if (textureWidth <= 0 || textureHeight <= 0) return cardPortraitAspectRatio
        return textureWidth.toFloat() / textureHeight.toFloat()
    }

    private fun angleDelta(
        value: Float,
        center: Float,
    ): Float {
        var delta = value - center
        while (delta > PI_FLOAT) delta -= TWO_PI_FLOAT
        while (delta < -PI_FLOAT) delta += TWO_PI_FLOAT
        return delta
    }

    private fun smoothStep(
        edge0: Float,
        edge1: Float,
        value: Float,
    ): Float {
        val width = (edge1 - edge0).coerceAtLeast(0.001f)
        val t = ((value - edge0) / width).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun lerp(
        a: Float,
        b: Float,
        t: Float,
    ): Float = a + (b - a) * t.coerceIn(0f, 1f)

    companion object {
        private const val FRAME_TIME_SECONDS = 0.016f
        private const val MAX_SENSOR_SHIFT = 0.5f
        private val PI_FLOAT = PI.toFloat()
        private val TWO_PI_FLOAT = (PI * 2.0).toFloat()
    }
}
