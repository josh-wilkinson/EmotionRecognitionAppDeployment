package com.example.facesdeployment

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.example.facesdeployment.ml.Extra
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {

    private var our_request_code : Int = 123

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
    }

    fun takePhoto(view : View) {
        // start intent to capture image
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        // check camera works
        if (intent.resolveActivity(packageManager) != null) {
            startActivityForResult(intent, our_request_code)
        }
    }

    // on activity start
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == our_request_code && resultCode == RESULT_OK) {
            val imageView : ImageView = findViewById(R.id.image)
            val bitmap = data?.extras?.get("data") as Bitmap

            var resize : Bitmap = Bitmap.createScaledBitmap(bitmap, 48, 48, true)

            imageView.setImageBitmap(resize)
            makeAGuess(resize)
        }
    }

    // converts rgb image to greyscale
    private fun getByteBuffer(bitmap: Bitmap): ByteBuffer? {
        val width = bitmap.width
        val height = bitmap.height
        val mImgData = ByteBuffer
            .allocateDirect(4 * width * height)
        mImgData.order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        for (pixel in pixels) {
            mImgData.putFloat(Color.red(pixel) / 255.0f)
        }
        return mImgData
    }

    fun makeAGuess(image : Bitmap) {
        val model = Extra.newInstance(applicationContext)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 48, 48, 1), DataType.FLOAT32)

        val byteBuffer = getByteBuffer(image)

        if (byteBuffer != null) {
            inputFeature0.loadBuffer(byteBuffer)
        }

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
        val confidences = outputFeature0.floatArray

        // find the index of the class with the biggest confidence.
        var maxPos = 0
        var maxConfidence = 0f
        for (i in confidences.indices) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i]
                maxPos = i
            }
        }

        val classes = arrayOf(
            "anger",
            "disgust",
            "fear",
            "happiness",
            "sadness",
            "suprise",
            "neutral"
        )

        Log.i("EMOTION", "" + classes[maxPos] + ", " + maxConfidence)

        val textView : TextView = findViewById(R.id.textView)
        textView.setText("" + classes[maxPos] + ", " + maxConfidence)

        // Releases model resources if no longer used.
        model.close()

    }
}