package com.example.alertsystem;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

import java.util.Locale;

public class MainActivity extends AppCompatActivity implements TextToSpeech.OnInitListener {
    EditText msg;
    Button speak_button;
    TextToSpeech TTS;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        TTS =  new TextToSpeech(this,this);
        msg=findViewById(R.id.message);
        speak_button=findViewById(R.id.speak_button);
        speak_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                speak();

            }

        });
    }

    @Override
    public void onInit(int i) {
        if(i==TextToSpeech.SUCCESS){
            int result= TTS.setLanguage(Locale.ENGLISH);
            TTS.setSpeechRate(1);
            TTS.setPitch(1);
            if(result== TextToSpeech.LANG_MISSING_DATA||result==TextToSpeech.LANG_NOT_SUPPORTED){
                Log.e("TTS","Language not supported");
            }
            else{
                speak_button.setEnabled(true);
                speak();
            }
        }
        else{
            Log.e("TextToSpeech","intilization failed");

        }

    }
    private void speak() {
        String msg="YOU HAVE ENTERED EMERGENCY AREA";
        TTS.speak(msg,TextToSpeech.QUEUE_FLUSH,null,null);
    }
}