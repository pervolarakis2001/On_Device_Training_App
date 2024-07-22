package com.example.appondevicetraining;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ProgressBar;

import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class ProgressBarAnimation extends Fragment {
    private ProgressBar progressBar;


    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.loading_page, container, false);
        progressBar = view.findViewById(R.id.progressBar2);
        return view;
    }

    
}
