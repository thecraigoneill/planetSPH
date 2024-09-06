#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define Max_particles 20000
#define kParticleMass 3.0e20
#define kH 7.89e5

#define Resolution_X 90
#define Resolution_Y 90
#define Resolution_Z 30

struct Particle
{
    double x[3];
    double u[3];

    double P;
    double density;
    double energy;
   // int Cell[2];
   // int Types;
};

int main(int argc, char *argv[])
{
    FILE *fp_in,*fp_out;
    int i,j;
    int l,m,n;
    int n_particles=0;
    double x_min[3],x_max[3],dx[3];
    struct Particle particles[Max_particles];
    double P[Resolution_X][Resolution_Y][Resolution_Z];
    double density[Resolution_X][Resolution_Y][Resolution_Z];
    double energy[Resolution_X][Resolution_Y][Resolution_Z];
    char temp[256];

   if (argc != 3 ) {

	printf("Must have two arguments - file_in and file_out.\n");
	exit(0);
   }

    fp_in=fopen(argv[1],"r");
    if(fp_in==NULL)
    {
        printf("Can't open file.\n");
        exit(0);
    }
    //fgets(temp,256,fp_in);
    //printf("%s\n",temp);
    printf("Working on file %s \n",argv[1]);

    for(i=0;i<Max_particles;i++)
    {
        if(feof(fp_in)!=0)
            break;
        fscanf(fp_in,"%le %le %le %le %le %le %le %le %le\n",&(particles[i].x[0]),&(particles[i].x[1]),&(particles[i].x[2]),
         //      &(particles[i].Cell[0]),&(particles[i].Cell[1]),&(particles[i].Types),
	       &(particles[i].P),&(particles[i].energy),&(particles[i].density),
               &(particles[i].u[0]),&(particles[i].u[1]),&(particles[i].u[2]));
        //if(i==0)
        //printf("%8e %8e %8e %8e %8e %8e %8e %8e %8e\n",particles[i].x[0],particles[i].x[1],particles[i].x[2],
        //       particles[i].P,particles[i].energy,particles[i].density,
        //       particles[i].u[0],particles[i].u[1],particles[i].u[2]);

        for(j=0;j<3;j++)
        {
            if(i==0)
            {
                for(j=0;j<3;j++)
                {
                    x_min[j]=particles[i].x[j];
                    x_max[j]=particles[i].x[j];
                }
            }
            else
            {
                if(x_min[j]>particles[i].x[j])x_min[j]=particles[i].x[j];
                if(x_max[j]<particles[i].x[j])x_max[j]=particles[i].x[j];
            }
        }
    }
    fclose(fp_in);
    n_particles=i;
    for(j=0;j<3;j++)
    {
        x_max[j]+=kH*2;
        x_min[j]-=kH*2;
    }
    printf("%d particles read.\n",n_particles);
    printf("[%8e %8e %8e] - [%8e %8e %8e]\n",x_min[0],x_min[1],x_min[2],x_max[0],x_max[1],x_max[2]);

    dx[0]=(x_max[0]-x_min[0])/(Resolution_X-1);
    dx[1]=(x_max[1]-x_min[1])/(Resolution_Y-1);
    dx[2]=(x_max[2]-x_min[2])/(Resolution_Z-1);

    for(l=0;l<Resolution_X;l++)
        for(m=0;m<Resolution_Y;m++)
            for(n=0;n<Resolution_Z;n++)
    {
        double xj[3];
        xj[0]=x_min[0]+dx[0]*l;
        xj[1]=x_min[1]+dx[1]*m;
        xj[2]=x_min[2]+dx[2]*n;
        density[l][m][n]=0;
        for(i=0;i<n_particles;i++)
        {
            double dxi[3],r2,r;
            double W,ratio;
            for(j=0;j<3;j++)
                dxi[j]=particles[i].x[j]-xj[j];
            r2=dxi[0]*dxi[0]+dxi[1]*dxi[1]+dxi[2]*dxi[2];
            r=sqrt(r2);
            W=0.;
            ratio=r/kH;
            if(ratio<=1.0)
            {
                double weight = 15/(M_PI*pow(kH,6));
                // For spiky kernel
                W = weight*pow((kH - r),3);
            }
            density[l][m][n] += kParticleMass * W;
        }
    }
    fp_out=fopen(argv[2],"w");
    printf("Output to file %s \n",argv[2]);
    fprintf(fp_out,"# vtk DataFile Version 2.0\n");
    fprintf(fp_out,"Grid output for SPH\n");
    fprintf(fp_out,"ASCII\n");
    fprintf(fp_out,"DATASET STRUCTURED_POINTS\n");
    fprintf(fp_out,"DIMENSIONS %d %d %d\n",Resolution_X,Resolution_Y,Resolution_Z);
    fprintf(fp_out,"ORIGIN %e %e %e\n",x_min[0],x_min[1],x_min[2]);
    fprintf(fp_out,"SPACING %e %e %E\n",dx[0],dx[1],dx[2]);
    fprintf(fp_out,"POINT_DATA %d\n",Resolution_X*Resolution_Y*Resolution_Z);
    fprintf(fp_out,"SCALARS Density double\n");
    fprintf(fp_out,"LOOKUP_TABLE default\n");
    for(n=0;n<Resolution_Z;n++)
        for(m=0;m<Resolution_Y;m++)
            for(l=0;l<Resolution_X;l++)
    {
        fprintf(fp_out,"%e\n",density[l][m][n]);
    }
    fclose(fp_out);
    return 0;
}
