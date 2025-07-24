clc; clear; close all;
    
    robot;

    % Definir el número de puntos para calcular las trayectorias
    num_points = 50;

    % Definir las posiciones articulares mínimas y máximas de acuerdo a los límites
    q_min = [-30, -45, -30, -135, -50]*pi/180;  % Posiciones mínimas en radianes
    q_max = [45, 5, 120, 5, 20]*pi/180;         % Posiciones máximas en radianes

    % Generar trayectorias articulares entre q_min y q_max usando jtraj
    q_traj1 = jtraj(q_min, q_max, num_points);  % Trayectoria desde posición mínima a máxima
    q_traj2 = jtraj(q_max, q_min, num_points);  % Trayectoria de vuelta (máxima a mínima)

    % Inicializar matrices para almacenar las posiciones en el espacio cartesiano
    matXY1 = zeros(num_points, 3);  % Para la vista XY
    matXY2 = zeros(num_points, 3);  % Para la vista XY (trayectoria de vuelta)
    matXZ1 = zeros(num_points, 3);  % Para la vista XZ

    % Crear la figura y el gráfico de las vistas 2D
    figure;

    % --- Vista 1: Gráfico en el plano XY ---
    subplot(1, 2, 1);
    title('Espacio de trabajo: vista XY');
    hold on;

    % Calcular las posiciones para la trayectoria 1 (q_traj1)
    for i = 1:num_points
        T = R.fkine(q_traj1(i, :));  % Cinemática directa para cada punto
        matXY1(i,:) = T.t;           % Guardar la posición en XY
    end

    % Calcular las posiciones para la trayectoria 2 (q_traj2)
    for i = 1:num_points
        T = R.fkine(q_traj2(i, :));  % Cinemática directa para cada punto
        matXY2(i,:) = T.t;           % Guardar la posición en XY
    end

    % Graficar las trayectorias en el plano XY
    plot(matXY1(:, 1), matXY1(:, 2), 'LineWidth', 2, 'Color', 'r');  % Trayectoria 1
    plot(matXY2(:, 1), matXY2(:, 2), 'LineWidth', 2, 'Color', 'b');  % Trayectoria 2
    xlabel('X');
    ylabel('Y');
    grid on;
    axis equal;

    % --- Vista 2: Gráfico en el plano XZ ---
    subplot(1, 2, 2);
    title('Espacio de trabajo: vista XZ');
    hold on;

    % Calcular las posiciones para la trayectoria 1 (q_traj1) en el plano XZ
    for i = 1:num_points
        T = R.fkine(q_traj1(i, :));  % Cinemática directa para cada punto
        matXZ1(i,:) = T.t;           % Guardar la posición en XZ
    end

    % Graficar la trayectoria en el plano XZ
    plot(matXZ1(:, 1), matXZ1(:, 3), 'LineWidth', 2, 'Color', 'g');
    xlabel('X');
    ylabel('Z');
    grid on;
    axis equal;

    hold off;