clear 
clc
close all

%matriz dh parametros
dh = [
    % tita   d     a    alfa   sigma
    0.000  0.160  0.160  pi/2  0.000; 
    -pi/2  0.080  0.000  -pi/2  0.000; 
    pi/2  0.000  0.440  0.000  0.000; 
    0.000  0.000  0.480  0.000  0.000; 
    -pi/2  0.000  0.270  0.000  0.000]; 

R = SerialLink(dh,'name','Atalante');

% Define the tool (the foot) at a distance below the ankle (e.g., 0.1 meter)
R.tool = transl(0 , 0, 0); 
R.base = transl(0,0,0)* trotx(pi/2);

% Define the initial joint positions
q = [0,0,0,0,0];

% Updated joint limits
R.qlim(1,1:2) = [-30,  45]*pi/180; % Aducción 0-30° Abducción 0-45°
R.qlim(2,1:2) = [-45,  5]*pi/180; % Rotación externa de cadera 0-45° y Rotación interna de pie 30°(considero 5° porque la articulacióne está en la cadera y no en el pie) 
R.qlim(3,1:2) = [-30,  120]*pi/180; % Flexión Muslo al torso 0-120° y extensión 0-30°
R.qlim(4,1:2) = [-135, 5]*pi/180; % Rodilla flexion 0-135° extensión 0-5°
R.qlim(5,1:2) = [-50,  20]*pi/180; % Dorsiflexión 0-20° Flexión Planar 0-50°

% Joint offset
R.offset = [0 pi/2 -pi/2 0 pi/2]; % Initial offset for the first hip rotation

% Robot visualization
R.plot(q, 'scale', 0.4, 'workspace', [-1.2 1.2 -1.2 1.2 -1.2 1.2]);

% Display the joint manipulation interface
R.teach();