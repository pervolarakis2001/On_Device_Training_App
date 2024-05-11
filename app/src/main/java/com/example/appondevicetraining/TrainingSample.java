package com.example.appondevicetraining;

import org.tensorflow.lite.support.image.TensorImage;

import java.util.List;

public class TrainingSample {
    private TensorImage image;
    private List<Float> label;

    public TrainingSample(TensorImage image,  List<Float> label) {
        this.image =image;
        this.label = label;
    }

    public TensorImage  getImage() {
        return image;
    }

    public  List<Float> getLabel() {
        return label;
    }
}
