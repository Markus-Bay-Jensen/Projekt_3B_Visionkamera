MODULE Module1
        CONST robtarget F:=[[100.87757472,-421.187709995,288.556653332],[0.000019631,-0.687657391,0.725926029,-0.01259814],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget B_1:=[[-24.682809342,-338.720179639,55.181617931],[0.000019654,-0.687657419,0.725926003,-0.012598092],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget B_2:=[[-17.397592482,-421.187704428,55.181448585],[0.000019651,-0.687657411,0.725926011,-0.01259811],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget G_1:=[[75.455258381,-338.72021178,55.181592946],[0.00001964,-0.687657403,0.725926017,-0.012598113],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget G_2:=[[100.877575351,-421.187711157,55.18143251],[0.000019637,-0.687657429,0.725925993,-0.012598133],[-1,0,-2,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget R_2:=[[-159.049884317,-421.187688774,55.181468243],[0.000019672,-0.687657395,0.725926026,-0.012598092],[-2,0,-3,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
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
        !Add your code here
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
ENDMODULE