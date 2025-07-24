clear 
clc
close all

% Matriz DH parámetros
dh = [
    % theta   d      a      alpha sigma
    0.000    0.160  0.160  pi/2   0.000; 
   -pi/2    0.080  0.000 -pi/2   0.000; 
    pi/2    0.000  0.440  0.000  0.000; 
    0.000   0.000  0.480  0.000  0.000; 
   -pi/2    0.000  0.270  0.000  0.000
];
%d es distancia a lo largo del eje z entre el sistema de coordenadas
%anterior y el actual
%a: distancia entre los ejes z consecutivos a lo largo del eje x del
%sistema de coordenadas del eslabón anterior


% Crear el objeto del robot usando la matriz DH
R = SerialLink(dh,'name','Atalante');

% Definir la herramienta (pie) a una distancia bajo el tobillo (0.1 metro)
R.tool = transl(0 , 0, 0); 
R.base = transl(0,0,0) * trotx(pi/2);

%%%%%%%% HAGO LA PARTE DE CINEMÁTICA DIRECTA PARA COMPARAR EN CINEMÁTICA INVERSA  %%%%

% Definir las posiciones articulares (convertidas a radianes)
q = [13.7, -19, 23.1, -65, -22.3] * pi/180;

% Límites articulares actualizados
R.qlim(1,1:2) = [-30,  45] * pi/180; % Aducción 0-30°, Abducción 0-45°
R.qlim(2,1:2) = [-45,  5]  * pi/180; % Rotación externa de cadera 0-45° y rotación interna
R.qlim(3,1:2) = [-30, 120] * pi/180; % Flexión muslo-torso 0-120°, extensión 0-30°
R.qlim(4,1:2) = [-135, 5]  * pi/180; % Rodilla flexión 0-135°, extensión 0-5°
R.qlim(5,1:2) = [-50,  20] * pi/180; % Dorsiflexión 0-20°, flexión plantar 0-50°

% Desplazamiento articular
R.offset = [0 pi/2 -pi/2 0 pi/2]; % Offset inicial

% Visualización del robot
R.plot(q, 'scale', 0.4, 'workspace', [-1.2 1.2 -1.2 1.2 -1.2 1.2]);

% Mostrar la interfaz para manipular las articulaciones
R.teach();

% Calcular la matriz de transformación homogénea para las articulaciones dadas
T = R.fkine(q)

% Asegurarse de que la matriz T sea numérica, si no hago esto me da error
T = double(T);
%T = invHomog(R.base) * T * invHomog(R.tool); %desacoplo la matriz T de la base y tool
% Mostrar la matriz de transformación
disp('Matriz de transformación homogénea T5_0:');
disp(T);


%%%%%%% PARTE DE CINEMÁTICA INVERSA %%%%%%%%%%%

% Calcular la cinemática inversa solo para la posición
%q = R.ikine(T, 'mask', [1 1 1 0 0 1]);
%Los primeros 3 1 indican que se debe tener en cuenta la posición (x, y, z).
%Los últimos 3 0 indican que se ignorará la orientación (roll, pitch, yaw).

% Mostrar el resultado de las configuraciones articulares
%disp('Configuraciones articulares q:');
%disp(q*180/pi);

% Graficar el robot en la configuración q obtenida
R.plot(q, 'scale', 0.2, 'workspace', [-1.2 1.2 -1.2 1.2 -1.2 1.2]);

% Multiplicar las matrices de transformación homogénea
T0_4 = T * transl(-dh(5,3), 0, 0); %Cuidado: los ejes están orientados igual que x5,y5,z5, no es T0_4 porque le falta la rotación, sirve para sacar P0_4
hold on;
trplot(T0_4, 'frame', '1', 'color', 'r', 'length', 0.1); % Muestra los ejes del sistema 1
%matriz del punto anterior al extremo del robot
disp('Matriz de transformación homogénea T4_0:');
disp(T0_4)

%Definicion del punto 4
    x4=T0_4(1,4)
    y4=T0_4(2,4)
    z4=T0_4(3,4)

  
    %Definicion del punto 0
    x0=0;
    y0=0;
    z0=0;

    %distancia R
    R=sqrt((z4-z0)^2+(x4-x0)^2)
    a1 = dh(1, 3)
    a2=sqrt((R)^2-(a1)^2)

    %angulo alfa
    alfa=acos((a2^2-a1^2-R^2)/(-2*a1*R))

    %angulo beta
    beta=atan2(z4-z0,x4-x0)

    %q1 (me da con 3,3 grados de diferencia del valor q1 propuesto (no se porque) asique le sumo eso en radianes, y q1 tiene solución única)
    q1=(alfa+beta+0.057451)%*180/pi
    q1GRADOS=q1*180/pi


    %calculo de matriz T0_1 (origen en 0 a 1)
    % Definir la base manualmente
    base = transl(0, 0, 0) * trotx(pi/2);
    T0_1 = base * trotz(q1) * transl(dh(1,3), 0, 0) * transl(0, 0, dh(1,2)) * trotx(dh(1,4))
   
    trplot(T0_1, 'frame', '1', 'blue', 'r', 'length', 0.1); % Muestra los ejes del sistema 1
    
    %calculo para q2, ángulo entre x1 y x2 alrededor de z1. Puedo
    %calcularlo entre x1 y x5 (dato) porque x2 es la misma orientación que
    %x5
    T0_5 = T; % T corresponde a T0_5 

    % Extraer los vectores x1 y x5 del plano xy
x1_xy = T0_1(1:2, 1); % Versor del eje x1 en el plano xy
x5_xy = T0_5(1:2, 1); % Versor del eje x5 en el plano xy (ya que x2 es paralelo a x5)

% Asegurarse de que los vectores están normalizados
x1_xy = x1_xy / norm(x1_xy);
x5_xy = x5_xy / norm(x5_xy);

% Calcular el ángulo entre los vectores usando el producto punto
cos_theta = dot(x1_xy, x5_xy);  % Producto punto

% El ángulo q2 en radianes
q2 = acos(cos_theta); % Este es el ángulo absoluto entre x1 y x5 en el plano xy

% Convertir el ángulo a grados
q2_deg = rad2deg(q2);

% Mostrar el ángulo
disp('El ángulo q2 entre x1 y x2:');
disp(q2_deg);

%%Una vez obtenidas las matrices homogeneas T4_0 y T2_0 ya quedan 3
%%articulaciones de rotación en el plano que se podrían resolver como en el
%%tp 5A

    


   
