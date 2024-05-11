package com.example.appondevicetraining;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Process;
import android.widget.Toast;

import androidx.fragment.app.Fragment;

/**
 * The sole purpose of this fragment is to request the necessary permissions.
 * It does not create a view.
 */
public class PermissionsFragment extends Fragment {

    private static final int PERMISSIONS_REQUEST_CODE = 10;
    private static final String[] PERMISSIONS_REQUIRED = {Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};

    private PermissionsAcquiredListener callback;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (!hasPermissions()) {
            requestPermissions(PERMISSIONS_REQUIRED, PERMISSIONS_REQUEST_CODE);
        } else {
            callback.onPermissionsAcquired();
        }
    }

    private boolean hasPermissions() {
        for (String permission : PERMISSIONS_REQUIRED) {
            if (getContext().checkPermission(permission, Process.myPid(), Process.myUid()) !=
                    PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSIONS_REQUEST_CODE) {
            boolean allPermissionsGranted = true;
            for (int grantResult : grantResults) {
                if (grantResult != PackageManager.PERMISSION_GRANTED) {
                    allPermissionsGranted = false;
                    break;
                }
            }
            if (allPermissionsGranted) {
                Toast.makeText(getContext(), "Permissions granted", Toast.LENGTH_LONG).show();
                callback.onPermissionsAcquired();
            } else {
                Toast.makeText(getContext(), "Permissions denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    public void setOnPermissionsAcquiredListener(PermissionsAcquiredListener callback) {
        this.callback = callback;
    }

    public interface PermissionsAcquiredListener {
        void onPermissionsAcquired();
    }
}
