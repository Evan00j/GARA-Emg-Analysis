#ifndef Hand_h
#define Hand_h

String NO_GRIP = "open"; 
String IN_GRIP = "grip";

typedef struct {                // Structure to hold and maintain hand information over time
  
  String gripState = NO_GRIP;
  
} hand;

#endif
