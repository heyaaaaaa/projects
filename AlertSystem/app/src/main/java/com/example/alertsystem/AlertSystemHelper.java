package com.example.alertsystem;

import android.app.PendingIntent;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.widget.Switch;

import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.location.Geofence;
import com.google.android.gms.location.GeofenceStatusCodes;
import com.google.android.gms.location.GeofencingRequest;
import com.google.android.gms.maps.model.LatLng;

public class AlertSystemHelper extends ContextWrapper {
private static final  String TAG="AlertSystemHelper";
PendingIntent pendingIntent;
    public AlertSystemHelper(Context base) {
        super(base);
    }

    public GeofencingRequest getGeofencingRequest(Geofence geofence){
      return new GeofencingRequest.Builder()
              .addGeofence(geofence)
              //.addGeofences()  for adding more
        .setInitialTrigger(GeofencingRequest.INITIAL_TRIGGER_ENTER)
              .build();
    }

    public  Geofence getGeofence(String ID , LatLng latLng,float radius,int transitionTypes){
       return  new Geofence.Builder()
           .setCircularRegion(latLng.latitude,latLng.longitude,radius)
       .setRequestId(ID)
               .setTransitionTypes(transitionTypes)
               .setLoiteringDelay(5000)
               //after how many seconds u want to be notified while moving in ambulance emergency area
    .setExpirationDuration(Geofence.NEVER_EXPIRE)
               .build();
    }

  public  PendingIntent getPendingIntent(){
        if(pendingIntent!=null){
            return  pendingIntent;
        }
      Intent intent= new Intent(this,AlertSystemBroadcastReceiver.class);
        pendingIntent=PendingIntent.getBroadcast(this,2607,intent,PendingIntent.
                FLAG_UPDATE_CURRENT);
        return  pendingIntent;
  }
  public  String getErrorString(Exception e){
        if(e instanceof ApiException){
            ApiException apiException=(ApiException) e;
            switch(apiException.getStatusCode()){
                case  GeofenceStatusCodes
                        .GEOFENCE_NOT_AVAILABLE:;
                return "GEOFENCE_NOT_AVAILABLE";
                case  GeofenceStatusCodes
                        .GEOFENCE_TOO_MANY_GEOFENCES:;
                    return "GEOFENCE_TOO_MANY_GEOFENCES";
                case  GeofenceStatusCodes
                        .GEOFENCE_TOO_MANY_PENDING_INTENTS:;
                    return "GEOFENCE_TOO_MANY_PENDING_INTENTS";
            }
        }
        return e.getLocalizedMessage();
  }
}
