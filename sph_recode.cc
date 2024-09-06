#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <assert.h>
#include <memory.h>
#include <omp.h>
using namespace std;

#define kPi 3.1415926535f
#define kParticleCount 20000
 
//#define kRestDensity 5520.0f
#define kRestDensity1 3.4e3f
#define kRestDensity2 3.4e3f
#define kRestDensity3 7.86e3f

#define kC 5000.0f
#define kRestEnergy1 1.6e11*1e-4   //erg/g ->*1e-7-> J/g -> *1e3 ->J/kg
#define kRestEnergy2 1.8e10*1e-4   //  erg/g -> J/kg

//Assuming constant volume is 1/10 of previous
#define kH 7.89e5
//#define kParticleMass 2.986e21
#define kParticleMass 3.0e20

 
//#define kDt ((1.0f/kFrameRate) / kSubSteps)
//timeStep in days
#define timeStep 0.0001
#define kDt timeStep*60*60*24
#define kDt2 (kDt*kDt)
// EndTime in years
#define EndSim 50
#define kEndTime EndSim*365.25*60*60*24
#define MEarth  5.97219e24
//#define kEndTime 0


#define Re 6371000.0
//Define rot in hours
#define kRot 2.5

 
 
struct Particle
{
    double x;
    double y;
    double z;
 
    double u;
    double v;
    double w;
	
    double ax;
    double ay;
    double az;
    double oax;
    double oay;
    double oaz;


    double P;
    double nearP;

    double h;
    double K1;
    double K2;
    double refden;
    double U0;
 
    float m;

    double txx;
    double txy;
    double txz;
    double tyy;
    double tyz;
    double tzz;

    double tdsdt;
    double dedt;

    int material;
 
    double density;
    float nearDensity;
    double energy;
};
 
struct Vector3
{
    Vector3() { }
    Vector3(double x, double y, double z) : x(x) , y(y), z(z) { }
    double x;
    double y;
    double z;
};

 
struct Rgba
{
    Rgba() { }
    Rgba(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) { }
    float r, g, b, a;
};
 
struct Material
{
    Material() { }
    Material(const Rgba& colour, float mass, float scale, float bias) : colour(colour) , mass(mass) , scale(scale) , bias(bias) { }
    Rgba colour;
    float mass;
    float scale;
    float bias;
};
 
#define kMaxNeighbourCount 5000
struct Neighbours
{
    const Particle* particles[kMaxNeighbourCount];
    float r[kMaxNeighbourCount];
    size_t count;
};
 
size_t particleCount = 0;
Particle particles[kParticleCount];
Neighbours neighbours[kParticleCount];
Vector3 prevPos[kParticleCount];
Vector3 relaxedPos[kParticleCount];
Material particleMaterials[kParticleCount];
Rgba shadedParticleColours[kParticleCount];

 
void writeVTK(size_t realTime)
{
    FILE *OUTPUT;
   // int t = time (NULL);
    time_t rawtime;
    struct tm * timeinfo;
    char myTime [80];
    char myTime2 [80];

    ///gettimeofday for millisecond
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(myTime,80,"%H%M%S",timeinfo);
   // strftime(myTime2, 80, "%C", realTime);
    char buffer[64];
    //snprintf(buffer, sizeof(char)*32, "out_%s.vtk", myTime);
    snprintf(buffer, sizeof(char)*32, "out_%.6lu.vtk", realTime);
    //OUTPUT = fopen("out1.vtk","w");
    OUTPUT = fopen(buffer,"w");
    fprintf(OUTPUT,"# vtk DataFile Version 2.0\n");
    fprintf(OUTPUT,"Time %s %lu \n",myTime, realTime);
    fprintf(OUTPUT,"ASCII\n");
    fprintf(OUTPUT,"DATASET UNSTRUCTURED_GRID\n");
    fprintf(OUTPUT,"POINTS %lu double\n",particleCount);
    
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
        fprintf(OUTPUT,"%.7e %.7e %7e \n",pi.x,pi.y,pi.z);
    }
    fprintf(OUTPUT,"\n");
    fprintf(OUTPUT,"CELLS %lu %lu\n",particleCount, 2*particleCount);
    for (size_t i=0; i<particleCount; ++i)
    {
        fprintf(OUTPUT,"1 %lu \n",i);
    }
    fprintf(OUTPUT,"\n");
    fprintf(OUTPUT,"CELL_TYPES %lu\n", particleCount);
    for (size_t i=0; i<particleCount; ++i)
    {
        fprintf(OUTPUT,"1 \n");
    }  
    fprintf(OUTPUT,"\n");  
    
    fprintf(OUTPUT,"POINT_DATA %lu\n", particleCount);
    

    fprintf(OUTPUT,"SCALARS Pressure double\n");
    fprintf(OUTPUT,"LOOKUP_TABLE default\n");
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
        fprintf(OUTPUT,"%f \n",pi.P);
    }  
    fprintf(OUTPUT,"\n");

    fprintf(OUTPUT,"SCALARS Energy double\n");
    fprintf(OUTPUT,"LOOKUP_TABLE default\n");
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
        fprintf(OUTPUT,"%f \n",pi.energy);
    }  
    fprintf(OUTPUT,"\n");

    fprintf(OUTPUT,"SCALARS Density double\n");
    fprintf(OUTPUT,"LOOKUP_TABLE default\n");
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
        fprintf(OUTPUT,"%f \n",pi.density);
    }  
    fprintf(OUTPUT,"\n");
    fprintf(OUTPUT,"VECTORS Velocity float\n");
    //fprintf(OUTPUT,"LOOKUP_TABLE default\n");
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
        fprintf(OUTPUT,"%f \t %f \t %f \n",pi.u,pi.v,pi.w);
    }  
    fprintf(OUTPUT,"\n");




    fclose(OUTPUT);
}


 
void ApplyBodyForces()
{
    #pragma omp parallel for
    for (int i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];

	pi.oax = pi.ax;
	pi.oay = pi.ay;
	pi.oaz = pi.az;

	pi.ax = 0.0;
	pi.ay = 0.0;
	pi.az = 0.0;

	 for (size_t j=0; j<particleCount; ++j)
    	 {
	   Particle& pj = particles[j];

	float sep_x = pj.x - pi.x;
	float sep_y = pj.y - pi.y;
	float sep_z = pj.z - pi.z;
	double dist = sqrt(sep_x*sep_x + sep_y*sep_y + sep_z*sep_z + 1e-2);
//	double distSixth = distSqr * distSqr * distSqr;  
  	double DistCube = dist*dist*dist;  
	//float rad_3 = sqrt(pow(sep_x,2) + pow(sep_y,2) + pow(sep_z,2));
 	//float distCube = rad_3 * rad_3 * rad_3;
	//float s = -pj.m*(6.67e-11)/ distCube;
	//float s = -pj.m*1e-6/ distCube;
	// Modify to calc mass from M = 4pi sum(r^2 * W) as per Benz 86 ??
//	float s = 
//	float s = 0;

	double G = 6.67e-11;

	if (dist > pi.h) {
	  float s = pj.m / DistCube;
	  pi.ax += sep_x * s * G;
	  pi.ay += sep_y * s * G;
	  pi.az += sep_z * s * G;
	  }
        //printf("Forces (%lu:%lu) X %le pj.x %le sep_x %le dist %0.12e pi.ax %.12e \n",i,j,pi.x,pj.x,sep_x,dist, pi.ax);

	}

//	double G1 = G*(MEarth/(kParticleCount*pi.m));
//	printf("Forces (%lu) X %le pi.ax %.12e \n",i,pi.x, pi.ax);

	// Vel Verlet velocities (single step)
	pi.u += 0.5*(pi.ax + pi.oax) * kDt; 
	pi.v += 0.5*(pi.ay + pi.oay) * kDt; 
	pi.w += 0.5*(pi.az + pi.oaz) * kDt; 

	float damping = 1.0; 
	pi.u *= damping;
	pi.v *= damping;
	pi.w *= damping;
	
//	if(i==1) {
//	printf("In forces, i=%lu, X=%le y=%le z=%le Vels: %le,%le,%le Acc=%le/%le/%le \n", i, pi.x, pi.y,pi.z,pi.u,pi.v,pi.w,pi.ax,pi.ay,pi.az);
//	}
//	pi.v -= 9.8f*kDt;
    }
}
 
 
void Advance()
{
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
 
	//if (i==1) {
	// printf("Advancing%lu): %.12e, Pos = %.12e PrevPos = %.12e kDt:%le U=%.12e \n",i,relaxedPos[i].x,pi.x,prevPos[i].x,kDt,pi.energy);
	//}

        // preserve current position
        prevPos[i].x = pi.x;
        prevPos[i].y = pi.y;
        prevPos[i].z = pi.z;
 

/* This needs a check on the new position in case something blows up... */

//	if ( x < 1e10 && y < 1e10 && z < 1e10) {
//	printf(" Advance: X %le y %le z %le \n",pi.x,pi.y,pi.z);
        pi.x += kDt * pi.u + 0.5*pi.ax*kDt*kDt;
        pi.y += kDt * pi.v + 0.5*pi.ay*kDt*kDt;
        pi.z += kDt * pi.w + 0.5*pi.az*kDt*kDt;
//	printf(" Advance (%lu): X %0.12e y %0.12e z %0.12e (previous: %.12le %.12le %.12le), Vel*dT: %le,%le,%le \n",i,pi.x,pi.y,pi.z,prevPos[i].x,prevPos[i].y,prevPos[i].z,pi.u*kDt,pi.v*kDt,pi.w*kDt);
//	}
//	if (i==1) {
//	 printf("Advanced(%lu): %.12e, Pos = %.12e PrevPos = %.12e kDt:%le U=%.12e \n",i,relaxedPos[i].x,pi.x,prevPos[i].x,kDt,pi.energy);
//	}

    }
}
 
 
void CalculatePressure()
{
    #pragma omp parallel for
    for (int i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];

        double density = 0;

	 neighbours[i].count = 0;
	 
	for (size_t j=0; j<particleCount; ++j)
    	   {
       		 Particle& pj = particles[j];

                    float dx = pj.x - pi.x;
                    float dy = pj.y - pi.y;
                    float dz = pj.z - pi.z;
			
                    float r2 = dx*dx + dy*dy + dz*dz;
		    float r = sqrt(r2);
  			
			// Using splines
	       		double W = 0.0;
	       		double ratio = r/pi.h;
		//	if (ratio <= 2.0) {

			// For splines
	       		//double weight = 1/(kPi*kH*kH*kH);

			// For spiky kernel
			double weight = 15/(kPi*pow(pi.h,6));
	       		if (ratio <= 1.0) {
				// For spiky
				W = weight*pow((pi.h - r),3); 
			
				// Poly6 kernel
				/*	double weight = 315/(64*kPi*pow(kH,9));
	       				if (ratio <= 1.0) {
					W = weight*pow((kH*kH - r*r),3); */
			
				// For splines
				//	W = weight * (1 - 1.5*ratio*ratio + 0.75*ratio*ratio*ratio);
			}


	       	//	else if (ratio > 1.0 && ratio < 2.0) {
		//		W = weight * 0.25*pow((2-ratio),3);
	       	//	}				
			// Splines Kernel
		
		       //double W = (1/(8*3.14*3.14*3.14))*exp(-r2/(kH*kH));
		       density += pj.m * W;
		       //printf("Mass: %le Density: %le W: %le kH/10  %lf < r %lf < kH20 %lf \n",pj.m, density, W, kH/10, r, kH*3);
		      if (neighbours[i].count < kMaxNeighbourCount)
                    	{
                        neighbours[i].particles[neighbours[i].count] = &pj;
                        neighbours[i].r[neighbours[i].count] = r;
			//printf("neighbours(%lu) countl %lu x: %le, r: %le \n",i,neighbours[i].count,pj.x,r);
                        ++neighbours[i].count;
                   	 }
		//	}
		   
             }
 	//density += pi.m * (1/(kH*sqrt(kPi)));  // are we sure about this?
        pi.density = density;
	//pi.h = kH * pow(pi.refden/pi.density,3);
	pi.h = kH;
	float a = 0.5;
	float b = 1.5;
	float A = 1.8e11*0.1; // (erg/cm3) * exp(-7) _> j/cm3 *exp(6) -> j/m3
	// This A should be bulk modulus, which is around 50e9 for granite
	float B = 1.8e11*0.1; //(erg/cm3) -> j/m3
	float nu = pi.density/pi.refden;
	float mu = nu -1;
	float alpha = 5;
	float beta = 5;
	
	//tillotson - condensed only
//	pi.P = (a + b/((pi.energy/(krestenergy1*(nu*nu))) + 1)) * pi.energy*pi.density + A*mu + B*mu*mu;
  //  	printf("tillotson %lu: p %.12e density %.12e nu %le mu %le \n",i,pi.p,pi.density,nu, mu);


	// murnaghan eos (isothermal...)
	// For MgO, K = 156GPa, n = 4.7
//	float k0 = 156e9;
//	float n = 4.7;
//	pi.P = (k0/n)* (pow(nu,n) - 1);
//	printf("Murnaghan %lu: P %.12e density %.12e nu %le mu %le \n",i,pi.P,pi.density,nu, mu);

//Tillotson
	if (pi.density > pi.refden || pi.energy < pi.U0) {
		//pi.P = (a + b/((pi.energy/(kRestEnergy1*(nu*nu))) + 1)) * pi.energy*pi.density + A*mu + B*mu*mu;
		pi.P = (a + b/((pi.energy/(pi.U0*(nu*nu))) + 1)) * pi.energy*pi.density + pi.K1*mu + pi.K2*mu*mu;
//	printf("Tillotson Condensed: %lu Pressure = %le  Energy = %le density = %le RestDensity = %lf RestEnergy = %lf \n", i, pi.P, pi.energy, pi.density, kRestDensity, kRestEnergy1);
	}
	else if (pi.density < pi.refden && pi.energy > kRestEnergy2) {
		pi.P = a*pi.energy*pi.density + ((b*pi.energy*pi.density/((pi.energy/(kRestEnergy2*nu*nu)) + 1)) + A*mu*exp(-alpha*((1/nu) - 1)))*exp(-B*((1/nu) - 1)*((1/nu) - 1));
//	printf("Tillotson expanded: %lu Pressure = %le  Energy = %le density = %le RestDensity = %lf RestEnergy = %lf \n", i, pi.P, pi.energy, pi.density, kRestDensity, kRestEnergy2);
	}
	else if (pi.energy > kRestEnergy1 && pi.energy < kRestEnergy2 && pi.density < pi.refden) {
		float Pc =  (a + b/((pi.energy/(kRestEnergy1*nu*nu)) + 1)) * pi.energy*pi.density + A*mu + B*mu*mu;
		float Pe = a*pi.energy*pi.density + ((b*pi.energy*pi.density/((pi.energy/(kRestEnergy2*nu*nu)) + 1)) + A*mu*exp(-alpha*((1/nu) - 1)))*exp(-B*((1/nu) - 1)*((1/nu) - 1));
		pi.P = (Pe*(pi.energy - kRestEnergy1) + Pc*(kRestEnergy2 - pi.energy))/(kRestEnergy2 - kRestEnergy1);
//	printf("Tillotson transitional: %lu Pressure = %le  Energy = %le density = %le RestDensity = %lf RestEnergy = %lf \n", i, pi.P, pi.energy, pi.density, kRestDensity, kRestEnergy2);

	}


	
    }
}
 
 
void CalculateRelaxedPositions()
{
   #pragma omp parallel for 
   for (int i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
 
        double x = pi.x;
        double y = pi.y;
        double z = pi.z;
	
	double u = 0.0;
	double v = 0.0;
	double w = 0.0;
	
	double accx = 0.0;
	double accy = 0.0;
	double accz = 0.0;



//	printf(" .. In calc pos, to start: %lf %lf %lf \n",x,y,z);

	float U=0.0;


	for (size_t j=0; j<particleCount; ++j)
    	   {
       		 Particle& pj = particles[j];

	pi.dedt = 0.0;
	pi.tdsdt = 0.0;
	pj.dedt = 0.0;
	pj.tdsdt = 0.0;

        //for (size_t j=0; j<neighbours[i].count; ++j)
        //{
          //  const Particle& pj = *neighbours[i].particles[j];
            //double r = neighbours[i].r[j];
            double dx = pi.x - pj.x;
            double dy = pi.y - pj.y;
            double dz = pi.z - pj.z;

	    double r2 = dx*dx + dy*dy + dz*dz;
	    double r = sqrt(r2);


	    double dx2 = sqrt(dx*dx);
            double dy2 = sqrt(dy*dy);
            double dz2 = sqrt(dz*dz);

	   //printf(" In calc pos (%lu,%lu): %le,%le,%le, pj:%le/%le/%le dx etc %le,%le,%le \n",i,j,x,y,z,pj.x,pj.y,pj.z,dx,dy,dz);		
	    //double W = (1/(kH*sqrt(kPi))) * exp(-(pow(r/kH,2)));
	   // Using splines
	       double W = 0.0;
	       double Wgradx = 0.0;
	       double Wgrady = 0.0;
	       double Wgradz = 0.0;
	       double ratio = r/pi.h;

	     // For spline kernel
	       double weight = 3/(2*kPi*kH*kH*kH);

	     // For spiky kernel
	//	double weight = 15/(kPi*pow(pi.h,6));
	       //double weight2 = r/(kPi*kH*kH*kH*kH*kH);
	       if (ratio <= 1.0) {
			//W = weight * (1 - 1.5*ratio*ratio + 0.75*ratio*ratio*ratio);
		//	Wgrad = weight*(-3*r/(kH*kH) + (9/4)*(r*r/(kH*kH*kH)));
			
			// For spiky kernel
	//		Wgradx = -3*(pi.h-r)*(pi.h-r)*(-dx/r)*weight;
	//		Wgrady = -3*(pi.h-r)*(pi.h-r)*(-dy/r)*weight;
	//		Wgradz = -3*(pi.h-r)*(pi.h-r)*(-dz/r)*weight;

			// For splines
			Wgradx = (weight*(-2.0 + (3.0/2.0)*ratio)/(kH*kH)) * (dx);
			Wgrady = (weight*(-2.0 + (3.0/2.0)*ratio)/(kH*kH)) * (dy);
			Wgradz = (weight*(-2.0 + (3.0/2.0)*ratio)/(kH*kH)) * (dz);
				}
	       else if (ratio > 1.0 && ratio < 2.0) {
		//	W = weight * 0.25*pow((2-ratio),3);
			//Wgrad = 0.25*weight*(-12/kH + 8*r/(kH*kH) - 3*(r*r/(kH*kH*kH)));
			Wgradx = (-1.0*weight*0.5*(2.0-ratio)*(2.0-ratio)/kH) * (dx)/r;
			Wgrady = (-1.0*weight*0.5*(2.0-ratio)*(2.0-ratio)/kH) * (dy)/r;
			Wgradz = (-1.0*weight*0.5*(2.0-ratio)*(2.0-ratio)/kH) * (dz)/r;
		       }	    

	    double Fdx=0.0;
	    double Fdy=0.0;
	    double Fdz=0.0;
 
	    double Fdu=0.0;
	    double Fdv=0.0;
	    double Fdw=0.0;

	    double fViscx = 0.0;
	    double fViscy = 0.0;
	    double fViscz = 0.0;
	    double hxx = 0.0;
	    double hxy = 0.0;
	    double hxz = 0.0;
	    double hyy = 0.0;
	    double hyz = 0.0;
	    double hzz = 0.0;
	    double hx = 0.0;
	    double hy = 0.0;
	    double hz = 0.0;
	    double he = 0.0;

            // viscocity
            double du = pi.u - pj.u;
            double dv = pi.v - pj.v;
            double dw = pi.w - pj.w;

	   // double LW = 45*(pi.h - r)/(kPi*pow(pi.h,6));

    	    // Changing sign here as per M92 and B86
	    /// This is Fvisc / Fbulk equation in B86 or Eq 4.1 or 4.2 in M92

            double u2 = du*dx + dv*dy + dw*dz;
	    double mu;
	    double I;
	    I = 0.0;
   	    mu = 1e3;
 	    if (u2 < 0)
            {
      		 mu = (pi.h * u2)/(r*r + 0.01*pi.h*pi.h);
		// beta = 2 for shocks, 0 for viscous accretion
		 avDens = (pi.density +pj.density)/2.0;
		 I = (1.0/avDens) * (-0.0*kC*mu + 2.0*mu*mu);
	//	printf(" IN here u %.12e mu %le I %.12e \n",u,mu,I);
	    }
	    else {
		mu = 0.0;
		I = 0.0;
	    }
	   
          /// Calc Fdrag

	  //if (r/kH < 4) {
	//	Fdrag_x = -5*(u2/sqrt(u2*u2)) * pi.u*W*(r/sqrt(r*r));
	//	Fdrag_y = -5*(u2/sqrt(u2*u2)) * pi.v*W*(r/sqrt(r*r));
	//	Fdrag_z = -5*(u2/sqrt(u2*u2)) * pi.w*W*(r/sqrt(r*r));

//	}
	  // printf(" IN here u %.12e kH %le r %.12e mu %.12e kC %le dens %.12e I %.12e \n",u,kH,r,mu,kC,pi.density,I);


	    // CO - replacing with M92 Eq3.3/ 3.4. Pressure gradient term.check grd of W term at end...
	    if (pi.h > 0 && pi.density != 0.0 && pj.density != 0.0)
            {
		 Fdx = -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density)) + I)*Wgradx;
		 Fdy = -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density)) + I)*Wgrady;
		 Fdz = -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density)) + I)*Wgradz;
		 
	 	// Eq 3.17 M92 for energy
	//	Fdu = (pj.u - pi.u)*( -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))))*Wgradx;
	//	Fdv = (pj.v - pi.v)*( -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))))*Wgrady;
	//	Fdw = (pj.w - pi.w)*( -(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))))*Wgradz;

	//	Fdu = -0.5 * (pj.m)*(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))) *du*Wgradx*1.0;
	//	Fdv = -0.5 * (pj.m)*(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))) *dv*Wgrady*1.0;
	//	Fdw = -0.5 * (pj.m)*(pi.P/((pi.density*pi.density)) + (pj.P/(pj.density*pj.density))) *dw*Wgradz*1.0;


		// Viscosity equation (Liu and Liu)
		/*fViscx = mu*(pj.u - pi.u)*(pj.m/pj.density)*LW;
		fViscy = mu*(pj.v - pi.v)*(pj.m/pj.density)*LW;
		fViscz = mu*(pj.w - pi.w)*(pj.m/pj.density)*LW; */

		hxx = 2*du*Wgradx - dv*Wgrady - dw*Wgradz;
		hxy = du*Wgrady + dv*Wgradx;
		hxz = du*Wgradz + dw*Wgradx;

		hyy = 2*dv*Wgrady - du*Wgradx - dw*Wgradz;
		hyz = dv*Wgradz + dw*Wgrady;
		
		hzz = 2*dw*Wgradz - du*Wgradx - dv*Wgrady;

		hxx *= (2/3)*hxx;
		hyy *= (2/3)*hyy;
		hzz *= (2/3)*hzz;

		pi.txx += pj.m * hxx /pj.density;
		pi.txy += pj.m * hxy /pj.density;
		pi.txz += pj.m * hxz /pj.density;
		pi.tyy += pj.m * hyy /pj.density;
		pi.tyz += pj.m * hyz /pj.density;
		pi.tzz += pj.m * hzz /pj.density;
 
		pj.txx += pi.m * hxx /pi.density;
		pj.txy += pi.m * hxy /pi.density;
		pj.txz += pi.m * hxz /pi.density;
		pj.tyy += pi.m * hyy /pi.density;
		pj.tyz += pi.m * hyz /pi.density;
		pj.tzz += pi.m * hzz /pi.density;

		
		fViscx = mu*(pi.txx/(pi.density*pi.density) + (pj.txx/(pj.density*pj.density)))*Wgradx + mu*(pi.txy/(pi.density*pi.density) + (pj.txy/(pj.density*pj.density)))*Wgrady + mu*(pi.txz/(pi.density*pi.density) + (pj.txz/(pj.density*pj.density)))*Wgradz;
		fViscy = mu*(pi.tyy/(pi.density*pi.density) + (pj.tyy/(pj.density*pj.density)))*Wgrady + mu*(pi.txy/(pi.density*pi.density) + (pj.txy/(pj.density*pj.density)))*Wgrady + mu*(pi.tyz/(pi.density*pi.density) + (pj.tyz/(pj.density*pj.density)))*Wgradz;
		fViscz = mu*(pi.tzz/(pi.density*pi.density) + (pj.tzz/(pj.density*pj.density)))*Wgradz + mu*(pi.txz/(pi.density*pi.density) + (pj.txz/(pj.density*pj.density)))*Wgradx + mu*(pi.tyz/(pi.density*pi.density) + (pj.tyz/(pj.density*pj.density)))*Wgrady;
 

	     // Add energy components
	    //Viscous entropy
	    pi.tdsdt = (pi.txx*pi.txx + 2.0*pi.txy*pi.txy + 2.0*pi.txz*pi.txz + pi.tyy*pi.tyy +2.0*pi.tyz*pi.tyz + pi.tzz*pi.tzz)*0.5*mu/pi.density;
	    hx = -1.0*(pi.P/(pi.density*pi.density) + pj.P/(pj.density*pj.density) + I)*Wgradx;	
	    hy = -1.0*(pi.P/(pi.density*pi.density) + pj.P/(pj.density*pj.density) + I)*Wgrady;	
	    hz = -1.0*(pi.P/(pi.density*pi.density) + pj.P/(pj.density*pj.density) + I)*Wgradz;	
	    he = (pj.u - pi.u)*hx +  (pj.v - pi.v)*hy + (pj.w - pi.w)*hz; 
	    pi.dedt += pj.m*he;	
//	printf("Relaxed(%i) tdsdt %le dedt %le he %le hx %le hy %le hz %le Wgradx %le\n",i,pi.tdsdt,pi.dedt,he,hx,hy,hz,Wgradx); 


//	if (r/kH < 2.0 & r > 0) {
//		double tmp = Fdx*kDt*kDt;
//		printf("%lu/%lu pi.den=%le pi.P=%le pj.P=%le Wgradx=%le weight=%le r=%lf I=%le Fdx=%.12e +x=%.12e \n",i,j, pi.density, pi.P, pj.P,Wgradx,weight,r,I,Fdx,tmp);
//	}
 	     }
	   else {
	    float Fdx=0.0;
	    float Fdy=0.0;
	    float Fdz=0.0;
  	    float Fdu=0.0;
	    float Fdv=0.0;
	    float Fdw=0.0;
	    fViscx = 0.0;
	    fViscy = 0.0;
	    fViscz = 0.0;

	//	printf("Hi!\n");

	   }
	
	    accx += (Fdx + pj.m*fViscx);
	    accy += (Fdy + pj.m*fViscy);
	    accz += (Fdz + pj.m*fViscz);

//            u += Fdx * kDt*damp;
//            v += Fdy * kDt*damp;
//            w += Fdz * kDt*damp;

	    //	printf(" ... in Relaxed pos x %le u %le t %le Fdx %le r %le acc %le \n",x,u,kDt,Fdx,r,accx); 


//	    x += u*kDt*damp;
///	    y += v*kDt*damp;
//	    z += w*kDt*damp;	
	
            
	    //U += (Fdu + Fdv + Fdw);     
         }

	
	float damp = 1.0;

	u = 0.5*(pi.ax+accx)*kDt*damp;
	v = 0.5*(pi.ay+accy)*kDt*damp;
	w = 0.5*(pi.az+accz)*kDt*damp;

/*	x += u*kDt;
 	y += v*kDt;
	z += w*kDt;	

        relaxedPos[i].x = x;
        relaxedPos[i].y = y;
        relaxedPos[i].z = z;
*/
	// float damp = 0.95;
	pi.u += u;
	pi.v += v;
	pi.w += w;

	x += pi.u*kDt + 0.5*accx*kDt*kDt;
	y += pi.v*kDt + 0.5*accy*kDt*kDt;
	z += pi.w*kDt + 0.5*accz*kDt*kDt;
	
        relaxedPos[i].x = x;
        relaxedPos[i].y = y;
        relaxedPos[i].z = z;
	
	pi.ax = accx;
	pi.ay = accy;
	pi.az = accz;

      	pi.energy += (pi.dedt*0.5 +pi.tdsdt)*kDt;
	if (pi.energy < 0.0) {
	   pi.energy = 0.0;
	}


    }
}
 
 
void MoveToRelaxedPositions()
{
    for (size_t i=0; i<particleCount; ++i)
    {
        Particle& pi = particles[i];
	
	  //printf("Move2Relax, x %le -> %le y %le -> %le z %le -> %le \n",pi.x,relaxedPos[i].x,pi.y,relaxedPos[i].y,pi.z,relaxedPos[i].z);

	  pi.x = relaxedPos[i].x;
          pi.y = relaxedPos[i].y;
          pi.z = relaxedPos[i].z;
    }
}
 
void CreateParticles(double& avDen, double& maxDen)
{

   float num = 0.0; 
   //particleCount = 0;
   int pCount = 0.0;
   double x[kParticleCount];
   double y[kParticleCount];
   double z[kParticleCount];
   double P[kParticleCount];
   double U[kParticleCount];
   double density[kParticleCount];
   double vx[kParticleCount];
   double vy[kParticleCount];
   double vz[kParticleCount];
   double minDen;
   maxDen = 1.0;
   minDen = 10000.0;


// Read file

    using namespace std;

    //ifstream input("one_planet.dat");
    //ifstream input("day_1.0hr_steady.dat");
   // ifstream input("stable_1.5hr_rot.dat");
  //  ifstream input("3_steadyEarth.dat"); 
   // ifstream input("randomEarth_20000.dat");
    ifstream input("input_77120_2_5_hr.dat");
    for (int i = 0; i< kParticleCount; i++) {
	input >> x[i];
	input >> y[i];
	input >> z[i];
	input >> P[i];
	input >> U[i];
	input >> density[i];
	input >> vx[i];
	input >> vy[i];
	input >> vz[i]; 

	
    }

    for (int i=0; i<kParticleCount; ++i)
        {
	Particle& pi = particles[particleCount];
	particleCount++;
		pi.x = x[i];
		pi.y = y[i];
		pi.z = z[i];
		pi.P = P[i];
		pi.energy = U[i];
		pi.density = density[i];
		if (pi.density > maxDen) {
			maxDen = pi.density;
		}
		if (pi.density < minDen) {
			minDen = pi.density;
		}

		pi.u = vx[i];
		pi.v = vy[i];
		pi.w = vz[i]; 
		// Values for chrondrite = olivine+iron / 2.
		pi.m = kParticleMass;
		pi.K1 = 131.0e9;
		pi.K2 = 75.0e9;
		pi.refden = kRestDensity1;
		pi.U0 = 550.0e6;
		pi.h = kH;
	   	pi.material = 0;
		pi.dedt = 0.0;
		pi.tdsdt = 0.0;

	}
	avDen = minDen + (maxDen - minDen)/2.0;
}
 
 
 
 
void CreateParticles4()
{

   float num = 0.0; 
   //particleCount = 0;
   int pCount = 0.0;
   double x[kParticleCount];
   double y[kParticleCount];
   double z[kParticleCount];
   double P[kParticleCount];
   double U[kParticleCount];
   double density[kParticleCount];
   double vx[kParticleCount];
   double vy[kParticleCount];
   double vz[kParticleCount];



// Read file

    using namespace std;

    //ifstream input("one_planet.dat");
    //ifstream input("day_1.0hr_steady.dat");
   // ifstream input("stable_1.5hr_rot.dat");
  //  ifstream input("3_steadyEarth.dat"); 
   // ifstream input("randomEarth_20000.dat");
    ifstream input("chrondrite_planet_24hrs_II.dat");
    //ifstream input("3_20000steady_vel_2hr.dat");
    for (int i = 0; i< kParticleCount; i++) {
	input >> x[i];
	input >> y[i];
	input >> z[i];
	input >> P[i];
	input >> U[i];
	input >> density[i];
	input >> vx[i];
	input >> vy[i];
	input >> vz[i]; 

	
    }

     double Mass = 0.0;
     double sumX = 0.0;
     double sumY = 0.0;
     double sumZ = 0.0;
     double comX = 0.0;
     double comY = 0.0;
     double comZ = 0.0;

    for (int i=0; i<kParticleCount; ++i)
        {
	Particle& pi = particles[particleCount];
	particleCount++;
		pi.x = x[i];
		pi.y = y[i];
		pi.z = z[i];
		pi.P = P[i];
		pi.energy = U[i];
		pi.density = density[i];
		pi.u = 0.0;
		pi.v = 0.0;
		pi.w = 0.0; 
		// Values for chrondrite = olivine+iron / 2.
		pi.m = kParticleMass;
		pi.K1 = 131.0e9;
		pi.K2 = 75.0e9;
		pi.refden = kRestDensity1;
		pi.U0 = 550.0e6;
		pi.material = 0;
		pi.dedt = 0.0;
		pi.tdsdt = 0.0;
		pi.h = kH;
		Mass += pi.m;
		sumX += pi.x*pi.m;
		sumY += pi.y*pi.m;
		sumZ += pi.z*pi.m;
		
	}
        if (Mass != 0.0) {
	comX = sumX/Mass;
	comY = sumY/Mass;
	comZ = sumZ/Mass;
	printf(" Centre of Mass: %le, %le, %le (compared with 5*Re=%le) \n",comX, comY, comZ,5*Re);
	}
        
    for (int i=0; i<kParticleCount; ++i)
        {
	Particle& pi = particles[i];
		if (kRot != 0.0) {
			double spin = kRot*60*60;

			pi.u = (pi.y - comY) * (2*kPi/spin);
			pi.v = -(pi.x - comX) * (2*kPi/spin);
			pi.w = 0.0;
		}
		else {
			pi.u = 0.0;
			pi.v = 0.0;
			pi.w = 0.0;
		}  
     }

}
 

void CreateParticles3()
{

   float num = 0.0; 
   //particleCount = 0;
   int pCount = 0.0;
   double x[kParticleCount];
   double y[kParticleCount];
   double z[kParticleCount];
   double P[kParticleCount];
   double U[kParticleCount];
   double density[kParticleCount];
 
// Read file

    using namespace std;

    //ifstream input("one_planet.dat");
   // ifstream input("day_1.2hr_steady.dat");
   // ifstream input("stable_1.5hr_rot.dat");
    ifstream input("randomEarth_20000.dat");
    for (int i = 0; i< kParticleCount; i++) {
	input >> x[i];
	input >> y[i];
	input >> z[i];
	input >> P[i];
	input >> U[i];
	input >> density[i];
	
    }

    for (int i=0; i<kParticleCount; ++i)
        {
	Particle& pi = particles[particleCount];
	particleCount++;
		pi.x = x[i];
		pi.y = y[i];
		pi.z = z[i];
		pi.P = P[i];
		pi.energy = U[i];
		pi.density = density[i];

		if (kRot != 0.0) {
			double spin = kRot*60*60;

			pi.u = (pi.y - 5*Re) * (2*kPi/spin);
			pi.v = -(pi.x - 5*Re) * (2*kPi/spin);
			pi.w = 0.0;
		}
		else {
			pi.u = 0.0;
			pi.v = 0.0;
			pi.w = 0.0;
		}  
		pi.m = kParticleMass;
		pi.K1 = 131.0e9;
		pi.K2 = 75.0e9;
		pi.refden = kRestDensity1;
		pi.U0 = 550.0e6;
		pi.material = 0;

	}
}


void CreateParticles2()
{

      float num = 0.0; 
     //particleCount = 0;
   int pCount = 0.0;
   double x[2000];
   double y[2000];
   double z[2000];

// Read file

    using namespace std;

    ifstream input("SI_planets.dat");
    for (int i = 0; i< 2000; i++) {
	input >> x[i];
	input >> y[i];
	input >> z[i];
    }

    for (int i=0; i<kParticleCount; ++i)
        {
	Particle& pi = particles[particleCount];
    	particleCount++;
		pCount = particleCount;
		num =  (0.03)*pCount   ;
		pi.x = x[i];
		pi.y = y[i];
		pi.z = z[i];
         //	pi.u = -0.6*(y[i] - (kViewHeight/2));
        // 	pi.v = 0.6*(x[i] - (kViewWidth/2));
        // 	pi.w = 0.0;
		pi.u = 0.0;
		pi.v = 0.0;
		pi.w = 0.0;
		pi.P = 0.0;
		pi.energy = 0.0;
		pi.density = 0.0;
		pi.m = kParticleMass;
		//pi.m = (4/3)*3.14*(kParticleRadius*kParticleRadius*kParticleRadius)*planet.material.mass;
		pi.energy = kRestEnergy1/10.0;
        	if (i > 1000) {
		   pi.u = -2100.0;
		   pi.v = -10.0;
		   pi.w = 0.0;
		}


	}
}

void CoreForm(size_t time, double& avDen, double& maxDen)
{
	size_t count1, count2;
	count1 = 0;
	count2 = 0;
	double t;
	double t_den;
	//t = (time - 1000.0)/2000.0;
	t = time/1000.0;
	//t_den = 5.5e3 + (1 - t)*(6.8e3 - 5.5e3);
	//t_den = 4.5e3 + (1 - t)*(5.7e3 - 4.5e3);
 	t_den = avDen + (1 - t)*(maxDen - avDen);
	printf(" ... time %lu t %lf t_den %le (MaxDen %le AvDen %le) \n",time,t,t_den,maxDen,avDen);
       for (int i=0; i<kParticleCount; ++i)
        {
		Particle& pi = particles[i];
		if (time == 0) {
		 	//if (pi.density < 5.5e3) {
		 	if (pi.density < 0.95*avDen) {
				pi.material = 0;
				count1++;
			}
			else {
				pi.material = 1;
				count2++;
			}
		}
		if (pi.material == 0) {
		//	if (pi.density < 10e3) {
			pi.K1 = 131e9;
			pi.K2 = 49e9;		//49e9
			pi.refden = 3200;
			pi.U0 = 650e6;
			/*  Original values
				pi.K1 = 131.0e9;
				pi.K2 = 75.0e9;
				pi.refden = kRestDensity1 (3400);
				pi.U0 = 550.0e6;
			*/
		//	}
		}
		else {
			if (pi.density > t_den || time > 950) {
			pi.K1 = 128e9;
			pi.K2 = 105e9; 
			pi.refden = 6000; //7860 - too high
			pi.U0 = 9.5e6; //9.5e6
			}
		}

	}
	printf("... Counts: Mantle: %lu Core %lu \n",count1, count2);


}

void CoreForm2()
{
       for (int i=0; i<kParticleCount; ++i)
        {
		Particle& pi = particles[i];
		if (pi.density > 5.5e3) {
		//	if (pi.density < 10e3) {
			pi.K1 = 128e9;
			pi.K2 = 105e9;
			pi.refden = 6860;
			pi.U0 = 9.5e6;
		//	pi.m *= 1.1;
		//	}
		}
		else {
		//	if (pi.density > 1.9e3) {
			pi.K1 = 131e9;
			pi.K2 = 49e9;
			pi.refden = 3400;
			pi.U0 = 550e6;

		//	pi.m *= 0.8;
		//	}
		}

	}


}



void readVTK() {
   double x[2000];
   double y[2000];
   double z[2000];
   double P[2000];
   double U[2000];
   double density[2000];

   char tmp[5000];
   string line;
// Read file

    using namespace std;

    int i = 0;
    int j = 0;
    int k = 0;
    int m = 0;
    int n = 0;
    ifstream input("condense_2_steady_planets/out_231257.vtk");
    while(input)
	{
	if (i < 5) {
	  //input >> tmp[i];
	  getline(input,line);
	  printf("tmp  is %s \n", line.c_str());
	}
	if (i>4 && i<2006) {
	input >> x[j];
	input >> y[j];
	input >> z[j];
	printf("i %d j %d X y Z %le %le %le \n",i,j,x[j],y[j],z[j]);
	++j;
	}
       	if (i >= 2006 && i< 6014) {
	  getline(input,line);
	  printf("%d tmp  is %s \n",i, line.c_str());
	}
	if (i>6013 && i<8014) {
	input >> P[k];
	printf("P %le  \n",P[k]);
	++k;
	}
	if (i > 8013 && i< 8017) {
	  //input >> tmp[i];
	  getline(input,line);
	  printf("tmp  is %s \n", line.c_str());
	}
	if (i>8016 && i<10017) {
	input >> U[m];
	printf("P %le  \n",U[m]);
	++m;
	}
	if (i > 10016 && i< 10020) {
	  //input >> tmp[i];
	  getline(input,line);
	  printf("tmp  is %s \n", line.c_str());
	}
	if (i>10019 && i<12020) {
	input >> density[n];
	printf("P %le  \n",density[n]);
	++n;
	}

	++i;
	}

//    for (int i = 0; i<5; ++i) {
//	input >> tmp[i];
  //  }
    //printf("Read header: %c\n",tmp);
 /*   for (int i = 0; i< 2000; ++i) {
	input >> x[i];
	input >> y[i];
	input >> z[i];
   }
   for (int i = 0; i< 4008; ++i) { //lines 2006-6010
	input >> tmp[i];
   }
   for (int i = 0; i < 2000; ++i) {
	input >> P[i];
   }
   for (int i = 0; i < 3; ++i) {
	input >> tmp[i];
   }
    for (int i = 0; i < 2000; ++i) {
	input >> U[i];
   }
   for (int i = 0; i < 3; ++i) {
	input >> tmp[i];
   }
    for (int i = 0; i < 2000; ++i) {
	input >> density[i];
   }
*/
  
   for (int i=0; i<particleCount; ++i)
        {
	Particle& pi = particles[i];
		pi.x = x[i];
		pi.y = y[i];
		pi.z = z[i];
       		pi.u = 0.0;
		pi.v = 0.0;
		pi.w = 0.0;
		pi.m = kParticleMass;
	//	pi.energy = U[i];
	//	pi.density = density[i];
	//	pi.P = P[i];
		if (i > 999) {
         	   pi.m = kParticleMass;
		   pi.u = -280.0;
		   pi.v = -200.0;
		   pi.w = 0.0;
		}

   }

}
 
 
int main (int argc, char** argv)
{
  /*  int tid,nthreads;
    char *cpu_name;        
        
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);
    cpu_name = (char *)calloc(80,sizeof(char));
*/
    memset(particles, 0, kParticleCount*sizeof(Particle));
    //readVTK();
    double avDen, maxDen;
    avDen = 0.0;
    maxDen = 0.0;
    CreateParticles(avDen, maxDen); 
    writeVTK(0);
    printf("Created vtk, \n");
    printf("MaxDen = %le AverageDen = %le \n",maxDen, avDen);
    size_t time;
    time = 0;
    size_t dens = 0;

    while (time < kEndTime) {
  
	ApplyBodyForces();
        Advance();
        CalculatePressure();
        CalculateRelaxedPositions();
        MoveToRelaxedPositions();
        writeVTK(time);

//	if (dens < 1 && time > 1000 && time < 2000 ) {
	if (dens < 1 && time < 1000) {
		CoreForm(time, avDen, maxDen);
		//dens = 1;
	}  
	else if (dens < 1 && time >= 1000) {
		dens = 1;
	}
	
	time += kDt;
        printf("Timestep: %lu sec \n",time);

    } 

//    MPI_Finalize(); 
    return 0;
}



