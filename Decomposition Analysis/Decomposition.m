x=zeros(5,3);  %4=number of countries (rows), 3=number of periods (bars)
y=zeros(5,3);  
z=zeros(5,3);
for i=1:5 % 4=number of countries (rows)
    for t=1:3 % 3=number of periods (bars)
        x0=worldS1(i,(t-1)*9+1);  %9=total number of years/columns of data available per country, 3=interval(e.g. 2010, 2055, 2100)
        y0=worldS1(i,27+(t-1)*9+1);
        z0=worldS1(i,54+(t-1)*9+1);

        dx=worldS1(i,t*9)-x0;
        dy=worldS1(i,27+t*9)-y0;
        dz=worldS1(i,54+t*9)-z0;
        
        x(i,t)=y0*z0*dx+1/2*dx*(z0*dy+y0*dz)+1/3*dx*dy*dz;
        y(i,t)=x0*z0*dy+1/2*dy*(z0*dx+x0*dz)+1/3*dx*dy*dz;
        z(i,t)=x0*y0*dz+1/2*dz*(y0*dx+x0*dy)+1/3*dx*dy*dz;
    end
end
