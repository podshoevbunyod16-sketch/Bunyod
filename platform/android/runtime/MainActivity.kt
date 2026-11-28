package ai.modelforge.modelforge_app

import android.app.ActivityManager
import android.content.Context
import android.os.Handler
import android.os.Looper
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.util.concurrent.Executors

class MainActivity : FlutterActivity() {
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "ai.modelforge/offline")
            .setMethodCallHandler { call, result ->
                if (call.method != "train" && call.method != "generate") {
                    result.notImplemented()
                    return@setMethodCallHandler
                }
                val payload = call.arguments as? String
                if (payload == null) {
                    result.error("INVALID_ARGUMENT", "Missing JSON payload", null)
                    return@setMethodCallHandler
                }
                executor.execute {
                    try {
                        if (!Python.isStarted()) Python.start(AndroidPlatform(applicationContext))
                        val info = ActivityManager.MemoryInfo()
                        (getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(info)
                        val reply = Python.getInstance().getModule("mobile_bridge")
                            .callAttr("request", call.method, filesDir.absolutePath, payload,
                                (info.availMem / (1024 * 1024)).toInt()).toString()
                        mainHandler.post { result.success(reply) }
                    } catch (e: Exception) {
                        mainHandler.post { result.error("OFFLINE_ERROR", e.message ?: "Local training failed", null) }
                    }
                }
            }
    }
}
