function CI = cin_inv(q0, T, bool)
    % Obtener parámetros DH del robot
    dh = [
        0.000  0.160  0.160  pi/2  0.000; 
    -pi/2  0.080  0.000  -pi/2  0.000; 
    pi/2  0.000  0.440  0.000  0.000; 
    0.000  0.000  0.480  0.000  0.000; 
    -pi/2  0.000  0.270  0.000  0.000];
    
    % Parámetros del objetivo
    x = T(1,4);
    y = T(2,4);
    z = T(3,4);  % Capturar el componente Z
    gamma = atan2(y, x);
    a2 = dh(3,3);
    a3 = dh(4,3);
    a4 = dh(5,3);

    % Definición de soluciones articulares
    q_1 = [q0(1), 0, 0, 0, 0];
    q_2 = [q0(1), 0, 0, 0, 0];

   
        % Ajuste para incluir z en el cálculo de x5 y y5
        r = sqrt(x^2 + y^2);  % Radio en el plano X-Y
        d = sqrt(r^2 + (z - dh(1,2))^2);  % Distancia efectiva considerando Z
        beta = atan2(z - dh(1,2), r);     % Ángulo de elevación en Z

        x5 = d * cos(beta) - a4 * cos(gamma);
        y5 = d * sin(beta) - a4 * sin(gamma);

        % Cálculo de soluciones articulares considerando z
        cos_q4 = ((x5^2) + (y5^2) - a2^2 - a3^2) / (2 * a2 * a3);
        cos_q4 = min(max(cos_q4, -1), 1);  % Limitar entre -1 y 1

        q_1(4) = acos(cos_q4);
        q_1(3) = atan2(y5, x5) - atan2(a3 * sin(q_1(4)), a2 + a3 * cos(q_1(4)));
        q_1(5) = gamma - q_1(4) - q_1(3);

        q_2(4) = -acos(cos_q4);
        q_2(3) = atan2(y5, x5) - atan2(a3 * sin(q_1(4)), a2 + a3 * cos(q_1(4)));
        q_2(5) = gamma - q_1(4) - q_1(3);

        % Ajuste de soluciones en rango [-pi, pi]
        q_1 = wrapToPi(q_1);
        q_2 = wrapToPi(q_2);

    % Selección de la mejor solución
    if bool
        if norm(q_1 - q0) < norm(q_2 - q0)
            CI = q_1;
        else
            CI = q_2;
        end
    else
        CI = [q_1; q_2];
    end
end
