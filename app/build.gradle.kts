import java.net.URI

plugins {
    alias(libs.plugins.androidApplication)
}

android {
    namespace = "com.example.appondevicetraining"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.appondevicetraining"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
//
    aaptOptions {
        noCompress; "tflite"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    buildFeatures {
        viewBinding = true
        mlModelBinding = true
    }


}


dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    implementation(libs.activity)
    implementation(libs.tensorflow.lite.support)
    implementation(libs.tensorflow.lite.metadata)
    implementation("org.tensorflow:tensorflow-lite:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.9.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.2")
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.9.0")
    implementation("org.tensorflow:tensorflow-core-platform:0.5.0")
    implementation("com.google.android.gms:play-services-tflite-java:16.0.1")
    implementation("com.google.android.gms:play-services-tflite-gpu:16.1.0")


    implementation("com.opencsv:opencsv:5.5.2")
    implementation("org.json:json:20201115")
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    // Kotlin lang
    implementation("androidx.core:core-ktx:1.8.0")
    implementation("androidx.fragment:fragment-ktx:1.4.1")

    // App compat and UI things
    implementation("androidx.appcompat:appcompat:1.5.0")
    implementation("com.google.android.material:material:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // Navigation library
    val nav_version = "2.5.1"
    implementation("androidx.navigation:navigation-fragment-ktx:$nav_version")
    implementation("androidx.navigation:navigation-ui-ktx:$nav_version")

    // CameraX core library
    val camerax_version = "1.2.0-alpha04"
    implementation("androidx.camera:camera-core:$camerax_version")

    // CameraX Camera2 extensions
    implementation("androidx.camera:camera-camera2:$camerax_version")

    // CameraX Lifecycle library
    implementation("androidx.camera:camera-lifecycle:$camerax_version")

    // CameraX View class
    implementation("androidx.camera:camera-view:$camerax_version")

    // WindowManager
    implementation("androidx.window:window:1.1.0-alpha03")

    // Unit testing
    testImplementation("junit:junit:4.13.2")

    // Instrumented testing
    androidTestImplementation("androidx.test.ext:junit:1.1.3")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.4.0")


}

