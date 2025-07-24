function [fqs, fds, f0s] = TD_PARK(fas, fbs, fcs, theta_r)

    fabcs = [fas; fbs; fcs];

    Ks = 2/3*[cos(theta_r) cos(theta_r - 2*pi/3) cos(theta_r + 2*pi/3)
          sin(theta_r) sin(theta_r - 2*pi/3) sin(theta_r + 2*pi/3) 
          1/2          1/2                   1/2];

    fqd0s = Ks * fabcs;

    fqs = fqd0s(1);
    fds = fqd0s(2);
    f0s = fqd0s(3);

end