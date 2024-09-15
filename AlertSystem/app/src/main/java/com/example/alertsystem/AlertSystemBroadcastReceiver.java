package com.example.alertsystem;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.location.Location;
import android.nfc.Tag;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.Toast;

import com.google.android.gms.location.Geofence;
import com.google.android.gms.location.GeofencingEvent;

import java.util.List;



import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.app.ActivityManagerCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.FragmentActivity;

import android.Manifest;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.location.Criteria;
import android.location.Location;
import android.location.LocationManager;
import android.nfc.Tag;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;







public class AlertSystemBroadcastReceiver extends BroadcastReceiver  {
    private static final String TAG = " AlertSystemBroadcastRe";
    TextToSpeech TTS;

    @Override
    //use async method if it cus  might kill our  broadcast receiver39;23
    public void onReceive(Context context, Intent intent) {
        // TODO: This method is called when the BroadcastReceiver is receiving
        // an Intent broadcast.
        //Toast.makeText(context, "fence triggered", Toast.LENGTH_SHORT).show();
NotificationHelper notificationHelper= new NotificationHelper(context);

        GeofencingEvent  geofencingEvent= GeofencingEvent.fromIntent(intent);
        if(geofencingEvent.hasError()){
            Log.d(TAG, "onReceive: error receiving geofence event...");
            return;
        }
        List<Geofence> geofenceList =geofencingEvent.getTriggeringGeofences();
        for(Geofence geofence: geofenceList){
            Log.d(TAG, "onReceive: "+geofence.getRequestId());
        }
     //   Location location=geofencingEvent.getTriggeringLocation();
        int transitionType=geofencingEvent.getGeofenceTransition();
        switch (transitionType){
            case Geofence.GEOFENCE_TRANSITION_ENTER:
                Toast.makeText(context, "YOU HAVE ENTERED COLLEGE ROAD", Toast.LENGTH_SHORT).show();

                TTS =  new TextToSpeech(context.getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int status) {
                        if (status == TextToSpeech.SUCCESS) {
                            String textToSay = "YOU HAVE ENTERED COLLEGE ROAD";
                            TTS.speak(textToSay, TextToSpeech.QUEUE_ADD, null,null);
                        }
                    }
                });
                notificationHelper.sendHighPriorityNotification("YOU HAVE ENTERED COLLEGE ROAD","",MapsActivity.class);
                break;
            case Geofence.GEOFENCE_TRANSITION_DWELL:
                Toast.makeText(context, "YOU ARE STILL IN COLLEGE ROAD", Toast.LENGTH_SHORT).show();
                TTS =  new TextToSpeech(context.getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int status) {
                        if (status == TextToSpeech.SUCCESS) {
                            String textToSay = "YOU ARE STILL IN COLLEGE ROAD";
                            TTS.speak(textToSay, TextToSpeech.QUEUE_ADD, null,null);
                        }
                    }
                });
                notificationHelper.sendHighPriorityNotification("YOU ARE STILL IN COLLEGE ROAD","",MapsActivity.class);
                break;
            case Geofence.GEOFENCE_TRANSITION_EXIT:
                Toast.makeText(context, "YOU HAVE LEFT COLLEGE ROAD", Toast.LENGTH_SHORT).show();
                TTS =  new TextToSpeech(context.getApplicationContext(), new TextToSpeech.OnInitListener() {
                    @Override
                    public void onInit(int status) {
                        if (status == TextToSpeech.SUCCESS) {
                            String textToSay = "YOU HAVE LEFT COLLEGE ROAD";
                            TTS.speak(textToSay, TextToSpeech.QUEUE_ADD, null,null);
                        }
                    }
                });notificationHelper.sendHighPriorityNotification("YOU HAVE LEFT COLLEGE ROAD","",MapsActivity.class);
                break;
        }
    }
}