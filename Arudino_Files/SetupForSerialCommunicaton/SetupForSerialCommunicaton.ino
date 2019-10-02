/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>

Servo index_finger;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 90;    // variable to store the servo position
int wind = 0;
void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
}

void updateGripState(){
  const char[] NO_GRIP = "open", MOVEMENT = "move", IN_GRIP = "grip";
  if (Serial.read() == "open") 
}

void loop() {
  
}
