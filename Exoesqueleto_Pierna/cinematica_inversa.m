% Longitudes de los eslabones
L1 = 1.0;
L2 = 0.5;
L3 = 0.3;

% Posición deseada
x = 0.8;
y = 0.4;
phi = pi/4;  % Orientación del extremo en radianes

% Calcular la cinemática inversa y animar el robot
[theta1, theta2, theta3] = cinematica_inversa_3gdl(x, y, phi, L1, L2, L3);

function [theta1, theta2, theta3] = cinematica_inversa_3gdl(x, y, phi, L1, L2, L3)
    % Cinemática inversa para un robot planar de 3 GDL
    % x, y: posición del extremo
    % phi: orientación del extremo (ángulo total)
    % L1, L2, L3: longitudes de los eslabones

    % Calcular la posición efectiva (x', y') para los primeros dos eslabones
    xe = x - L3 * cos(phi);  % Posición efectiva en x
    ye = y - L3 * sin(phi);  % Posición efectiva en y

    % Calcular theta2
    cos_theta2 = (xe^2 + ye^2 - L1^2 - L2^2) / (2 * L1 * L2);
    
    % Verificar si la solución es posible (cos_theta2 debe estar en el rango [-1, 1])
    if abs(cos_theta2) > 1
        error('No hay solución para esta posición.');
    end
    
    % Obtener las dos posibles soluciones para theta2
    theta2_1 = atan2(sqrt(1 - cos_theta2^2), cos_theta2);  % Codo arriba
    theta2_2 = atan2(-sqrt(1 - cos_theta2^2), cos_theta2);  % Codo abajo
    
    % Calcular theta1 para cada solución
    theta1_1 = atan2(ye, xe) - atan2(L2 * sin(theta2_1), L1 + L2 * cos(theta2_1));
    theta1_2 = atan2(ye, xe) - atan2(L2 * sin(theta2_2), L1 + L2 * cos(theta2_2));
    
    % Calcular theta3 (ángulo restante para la orientación)
    theta3_1 = phi - (theta1_1 + theta2_1);
    theta3_2 = phi - (theta1_2 + theta2_2);
    
    % Mostrar las dos soluciones
    theta1 = [theta1_1, theta1_2];
    theta2 = [theta2_1, theta2_2];
    theta3 = [theta3_1, theta3_2];

    % ---- Graficar el robot y animar el movimiento ----
    
    % Definir los parámetros DH del robot
    L(1) = Link([0 L1 0 pi/2], 'standard');
    L(2) = Link([0 0 L2 0], 'standard');
    L(3) = Link([0 0 L3 0], 'standard');
    
    % Crear el robot
    R = SerialLink(L, 'name', 'Robot3GDL');

    % Definir el workspace
    workspace = [-2 2 -2 2 -2 2];

    % Posiciones articulares para la solución 1 (codo arriba)
    q1 = [theta1_1, theta2_1, theta3_1];

    % Posiciones articulares para la solución 2 (codo abajo)
    q2 = [theta1_2, theta2_2, theta3_2];
    
    % Graficar el robot en la configuración inicial
    figure;
    R.plot(q1, 'workspace', workspace);
    title('Movimiento del Robot - Codo Arriba');

    % Animar el movimiento del robot (solución 1)
    figure(1);
    steps = 50;  % Cantidad de pasos para la animación
    q_traj = jtraj([0 0 0], q1, steps);  % Trajectoria desde posición neutra a q1
    R.plot(q_traj, 'workspace', workspace);
    title('Animación - Solución 1 Codo Arriba');

    % Graficar el robot en la configuración de codo abajo
    figure(2);
    R.plot(q2, 'workspace', workspace);
    title('Movimiento del Robot - Codo Abajo');

    % Animar el movimiento del robot (solución 2)
    figure(3);
    q_traj = jtraj([0 0 0], q2, steps);  % Trajectoria desde posición neutra a q2
    R.plot(q_traj, 'workspace', workspace);
    title('Animación - Solución 2 (Codo Abajo)');
end






