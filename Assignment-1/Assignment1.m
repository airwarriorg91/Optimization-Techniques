%Question1
clc; clear;
lb1=0; ub1=80*pi/180; %Constraints on theta
u=90; h=50; %Initial conditions of objective function
options=optimoptions("fmincon","Display","iter","TolFun",1e-8,"TolX",1e-8,"MaxIter",10000);
fun1=@(x)obj1(x,u,h);
theta0=45*pi/180;
[theta,fval]=fmincon(fun1,theta0,[],[],[],[],lb1,ub1,[],options);

theta=theta*180/pi; D=-fval;

%%
%Question3
clc; clear;
A=[1,0,1,0,1,0;0,1,0,1,0,1;1,1,0,0,0,0;0,0,1,1,0,0;0,0,0,0,1,1];
B=[460;560;200;310;420];
lb=[0,0,0,0,0,0];
options=optimoptions("fmincon","Display","iter","TolFun",1e-8,"TolX",1e-8,"MaxIter",10000);

%Variables defined as vector x as:
% S1PA,S1PB,S2PA,S2PB,S3PA,S3PB
x0=[460,560,200,310,420,150];
fun=@(x)obj3(x);

[vector,fval]=fmincon(fun,x0,A,B,[],[],lb,[],[],options);
supply1_all=vector(1:2); supply2_all=vector(3:4); supply3_all=vector(5:6);
profit=-fval;
%%
%Question9
clc; clear
A=[10,12,8]; B=2000;
lb=[0,0,0];
Aeq=[1,1,1]; Beq=20;
options=optimoptions("fmincon","Display","iter","TolFun",1e-8,"TolX",1e-8,"MaxIter",10000);

%Variables are: acres of land assingned for crops X, Y and Z.
x0=[1,1,1];
fun=@(x)obj9(x);
[vector,fval]=fmincon(fun,x0,A,B,Aeq,Beq,lb,[],[],options);
profit=-fval;

%%
%Question10
clc; clear;
lb=0; ub=105;
options=optimoptions("fmincon","Display","iter","TolFun",1e-8,"TolX",1e-8,"MaxIter",10000);
x0=10;
fun=@(x)obj10(x);
[x,volume]=fmincon(fun,x0,[],[],[],[],lb,ub,[],options);
volume=-volume*1e-6; %cubic metre

%%
%Question7
clc; clear;
A=[-1,0,0,-1,0,0,-1,0,0;1,1,1,0,0,0,0,0,0;0,0,0,1,1,1,0,0,0;0,0,0,0,0,0,1,1,1;(1-0.2),0,0,-0.2,0,0,-0.2,0,0;(-1+0.1),0,0,0.1,0,0,0.1,0,0;0,0,0.3,0,0,(-1+0.3),0,0,0.3;0,0,-0.35,0,0,(1-0.35),0,0,-0.35;
0,0.7,0,0,-(1-0.7),0,0,-(1-0.7),0];
B=[-5000;8000;10000;9000;0;0;0;0;0];
lb=[0,0,0,0,0,0,0,0,0];
options=optimoptions("fmincon","Display","iter","TolFun",1e-8,"TolX",1e-8,"MaxIter",10000);

%Variables defined as xij: weight of ith component used in making the jth
%blend, i=1,2,3 and j=1,2,3
fun=@(x)obj7(x);
x0=[1000,500,500,1000,500,500,1000,500,500];
[vector,fval]=fmincon(fun,x0,A,B,[],[],lb,[],[],options);
profit=-fval;

%Weight of ith bean used in blends 1,2 and 3:
bean1=vector(1:3); bean2=vector(4:6); bean3=vector(7:9);
clear vector
%%
function F=obj1(x,u,h)
g=9.81;
F=-u*cos(x)*((u/g)*sin(x)+sqrt(2*(h/g)+((u/g)*(u/g)*sin(x)*sin(x))));
end

function F=obj3(x)
F=-50000*(x(1)+x(2)+x(3)+x(4)+x(5)+x(6))+(1100*(x(1)+x(2))+1000*(x(3)+x(4))+900*(x(5)+x(6))+3000*x(1)+3500*x(2)+2000*x(3)+2500*x(4)+6000*x(5)+4000*x(6)+26000*(x(1)+x(3)+x(5))+21000*(x(2)+x(4)+x(6)));
end

function F=obj9(x)
F=10*(200*x(1)+300*x(2)+100*x(3))+40*(10*x(1)+12*x(2)+8*x(3))-8000*x(1)-9000*x(2)-5000*x(3);
end

function F=obj10(x)
F=-x*(210-2*x)*(297-2*x);
end

function F=obj7(x)
F=120*(x(1)+x(2)+x(3))+130*(x(4)+x(5)+x(6))+110*(x(7)+x(8)+x(9))-300*(x(1)+x(4)+x(7))-320*(x(2)+x(5)+x(8))-280*(x(3)+x(6)+x(9));
end