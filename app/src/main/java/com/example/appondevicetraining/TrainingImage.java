package com.example.appondevicetraining;

import android.graphics.Bitmap;

public class TrainingImage {
    private Bitmap bitmap;
    private String className;

    public TrainingImage(Bitmap bitmap, String className) {
        this.bitmap = bitmap;
        this.className = className;
    }

    public Bitmap getBitmap() {
        return bitmap;
    }

    public String getClassName() {
        return className;
    }
}