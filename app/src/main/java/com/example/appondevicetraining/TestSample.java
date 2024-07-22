package com.example.appondevicetraining;

import org.tensorflow.lite.support.image.TensorImage;

public class TestSample {
    private int label;
    private TensorImage tensorImage;

    public TestSample(int label, TensorImage tensorImage) {
        this.label = label;
        this.tensorImage = tensorImage;
    }

    public int getLabel() {
        return label;
    }

    public TensorImage getTensorImage() {
        return tensorImage;
    }
}
