MODULE Module1
    CONST robtarget F:=[[100.87757472,-420,200],[0.000019631,-0.687657391,0.725926029,-0.01259814],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget B_1:=[[50,-320,55],[0.000019654,-0.687657419,0.725926003,-0.012598092],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget B_2:=[[50,-420,55],[0.000019651,-0.687657411,0.725926011,-0.01259811],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget G_1:=[[100.455258381,-320,55],[0.00001964,-0.687657403,0.725926017,-0.012598113],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget G_2:=[[100,-420,55],[0.000019637,-0.687657429,0.725925993,-0.012598133],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget R_1:=[[-150,-320,55],[0.000019646,-0.687657503,0.725925923,-0.0125981],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget R_2:=[[-150,-420,55],[0.000019672,-0.687657395,0.725926026,-0.012598092],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    VAR socketdev temp_socket;
    VAR num received_nr:=9;
    VAR num received_nr2:=0;
    VAR bool ok := FALSE;
    VAR socketdev client_socket;
    VAR string received_string:="9";
    
    VAR string sent_string := "No Data Available";
    VAR bool stop_program := TRUE;

!***********************************************************
    !
    ! Module:  Module1
    !
    ! Description:
    !   <Insert description here>
    !
    ! Author: Markus Bay Jensen
    !
    ! Version: 1.0
    !
    !***********************************************************
    
    
    !***********************************************************
    !
    ! Procedure main
    !
    !   This is the entry point of your program
    !
    !***********************************************************
    PROC main()
        
        WHILE TRUE DO
            
        ENDWHILE
    ENDPROC
    PROC Path_10()
        MoveL F,v1000,z100,tool0\WObj:=wobj0;
        MoveL B_1,v1000,z100,tool0\WObj:=wobj0;
        MoveL B_2,v1000,z100,tool0\WObj:=wobj0;
    ENDPROC
    PROC Path_20()
        MoveL F,v1000,z100,tool0\WObj:=wobj0;
        MoveL G_1,v1000,z100,tool0\WObj:=wobj0;
        MoveL G_2,v1000,z100,tool0\WObj:=wobj0;
    ENDPROC
    PROC Path_30()
        MoveL F,v1000,z100,tool0\WObj:=wobj0;
        MoveL [[-159.117775873,-338.720206102,55.181753055],[0.000019646,-0.687657503,0.725925923,-0.0125981],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]],v1000,z100,tool0\WObj:=wobj0;
        MoveL R_2,v1000,z100,tool0\WObj:=wobj0;
    ENDPROC
    
    PROC op_Kummonikation()
       SocketCreate temp_socket;
       SocketBind temp_socket, "192.168.125.1", 53;
       SocketListen temp_socket;
        
    ENDPROC
    
    PROC Kummonikation()
           !Communication 
           SocketAccept temp_socket,client_socket,\Time:=86400;
           SocketReceive client_socket \list:=received_string;
           ok:= StrToVal(received_string,received_nr);
           SocketSend client_socket \Str:=sent_string;
           SocketClose client_socket;
           
    ENDPROC
    
    PROC ned_Kummonikation()
        
        
        SocketClose temp_socket;
           
           
    ENDPROC

ENDMODULE
ToStr