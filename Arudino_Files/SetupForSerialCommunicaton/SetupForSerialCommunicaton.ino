#include <Servo.h>
#include "Hand.h"

Servo index_finger,                 //Servo Object Creation
      middle_fingers, 
      pinky_finger, 
      thumb;  

hand GARA;
int pos = 0;

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(75);
  pinMode(13, OUTPUT);
  index_finger.attach(9);
  
}


void updateGripState(hand *hand){
  if(hand->gripState == NO_GRIP)
    hand->gripState = IN_GRIP;
  else hand->gripState = NO_GRIP;
  Serial.println("Status Updated!");
}

void doGripMotion(hand *hand){
  Serial.println("Moving");     //Send a movement keyword to stop pi from sending more commands
  if(hand->gripState == IN_GRIP){
   for (pos = 0; pos <= 180; pos += 1) { 
    index_finger.write(pos);                
    delay(15);                       
    }
  }
  if(hand->gripState == NO_GRIP){
    for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
      index_finger.write(pos);             
      delay(15);                       
    }
  }                                //Do movement code  here simulated by a delay for now
  Serial.println("Done"); 
  updateGripState(hand);          // Let Pi know its ready to move again
        
}




void loop() {
  int sensorValue = analogRead(A0);
  float voltage = sensorValue * (5.0 / 1023.0);
  Serial.println(voltage);
  
  if(Serial.readString() == "changeGrip"){
    doGripMotion(&GARA);
    Serial.println(GARA.gripState);
  }
  
  

  
}
