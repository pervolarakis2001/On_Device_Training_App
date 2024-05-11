package com.example.appondevicetraining;

import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;

import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.appondevicetraining.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;
import android.content.Intent;
public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView((R.layout.activity_main));
        if (savedInstanceState != null) {
            return;
        }
        PermissionsFragment firstFragment = new PermissionsFragment();
        getSupportFragmentManager()
                .beginTransaction()
                .add(R.id.fragmentContainerView, firstFragment)
                .commit();


    }

    @Override
    public void onAttachFragment(Fragment fragment) {
        if (fragment instanceof PermissionsFragment) {
            ((PermissionsFragment) fragment).setOnPermissionsAcquiredListener(() -> {
                CameraFragment cameraFragment = new CameraFragment();
                Log.d(TAG, "Before fragment replacement");

                getSupportFragmentManager()
                        .beginTransaction()
                        .replace(R.id.fragmentContainerView, cameraFragment)
                        .commit();
                Log.d(TAG, "after fragment replacement");
            });
        }
    }

    }