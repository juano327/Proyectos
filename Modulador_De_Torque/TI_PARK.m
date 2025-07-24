function [fas, fbs, fcs] = TI_PARK(fqs, fds, f0s , theta_r)

    fqd0s = [fqs; fds; f0s];

    Ks = [cos(theta_r)         sin(theta_r)             1
          cos(theta_r - 2*pi/3) sin(theta_r - 2*pi/3)    1
          cos(theta_r + 2*pi/3) sin(theta_r + 2*pi/3)    1];

    fabcs = Ks * fqd0s;

    fas = fabcs(1);
    fbs = fabcs(2);
    fcs = fabcs(3);

end