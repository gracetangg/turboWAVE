#include "meta_base.h"
#include "computeTool.h"
#include "elliptic.h"

//////////////////////////////////
//                              //
// ELLIPTICAL SOLVER BASE CLASS //
//                              //
//////////////////////////////////

// Fixed boundary values are expected to be loaded into the far ghost cells before calling Solve

EllipticSolver::EllipticSolver(const std::string& name,MetricSpace *m,Task *tsk) : BoundedTool(name,m,tsk)
{
	coeff = NULL;
	x0 = x1 = y0 = y1 = z0 = z1 = tw::bc::fld::natural;
	gammaBeam = 1.0;
}

void EllipticSolver::FormOperatorStencil(std::valarray<tw::Float>& D,tw::Int i,tw::Int j,tw::Int k)
{
	tw::Float kx = space->Dim(1)>1 ? 1.0 : 0.0;
	tw::Float ky = space->Dim(2)>1 ? 1.0 : 0.0;
	tw::Float kz = space->Dim(3)>1 ? 1.0 : 0.0;
	tw::Float Vol = space->dS(i,j,k,0);
	D[1] = kx*(space->dS(i,j,k,1)/Vol) / space->dl(i,j,k,1);
	D[2] = kx*(space->dS(i+1,j,k,1)/Vol) / space->dl(i+1,j,k,1);
	D[3] = ky*(space->dS(i,j,k,2)/Vol) / space->dl(i,j,k,2);
	D[4] = ky*(space->dS(i,j+1,k,2)/Vol) / space->dl(i,j+1,k,2);
	D[5] = kz*(space->dS(i,j,k,3)/Vol) / space->dl(i,j,k,3);
	D[6] = kz*(space->dS(i,j,k+1,3)/Vol) / space->dl(i,j,k+1,3);
	if (coeff!=NULL)
	{
		D[1] *= 0.5*((*coeff)(i-1,j,k) + (*coeff)(i,j,k));
		D[2] *= 0.5*((*coeff)(i,j,k) + (*coeff)(i+1,j,k));
		D[3] *= 0.5*((*coeff)(i,j-1,k) + (*coeff)(i,j,k));
		D[4] *= 0.5*((*coeff)(i,j,k) + (*coeff)(i,j+1,k));
		D[5] *= 0.5*((*coeff)(i,j,k-1) + (*coeff)(i,j,k));
		D[6] *= 0.5*((*coeff)(i,j,k) + (*coeff)(i,j,k+1));
	}
	D[0] = -(D[1] + D[2] + D[3] + D[4] + D[5] + D[6]);
}

void EllipticSolver::FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential)
{
	#pragma omp parallel
	{
		for (auto cell : EntireCellRange(*space))
			if (theRegion->Inside(space->Pos(cell),*space))
				phi(cell) = thePotential;
	}
}

void EllipticSolver::SetCoefficients(ScalarField *coefficients)
{
	coeff = coefficients;
}

void EllipticSolver::ZeroModeGhostCellValues(tw::Float *phi0,tw::Float *phiN1,ScalarField& source,tw::Float mul)
{
	// specialized for uniform grid in z
	// assumes source has already been resolved into transverse eigenmodes, and k=0 mode has i=j=1
	// computes ghost cell potentials for k=0 mode by integrating against discretized Green's function
	// the assumption is that either z0 or z1 or both are "natural"
	// if either z0 or z1 (but not both) are dirichletCell, first assume both are natural, then shift result appropriately

	tw::Int k,kg,kN1;

	*phi0 = 0.0;
	*phiN1 = 0.0;
	kN1 = task->globalCells[3]+1;

	for (k=1;k<=source.Dim(3);k++)
	{
		kg = task->GlobalCellIndex(k,3);
		*phi0 += 0.5*mul*sqr(dz(source))*source(1,1,k)*fabs(tw::Float(kg));
		*phiN1 += 0.5*mul*sqr(dz(source))*source(1,1,k)*fabs(tw::Float(kN1-kg));
	}

	task->strip[3].AllSum(phi0,phi0,sizeof(tw::Float),0);
	task->strip[3].AllSum(phiN1,phiN1,sizeof(tw::Float),0);

	if (z0==tw::bc::fld::dirichletCell)
	{
		*phiN1 -= *phi0;
		*phi0 = 0.0;
	}

	if (z1==tw::bc::fld::dirichletCell)
	{
		*phi0 -= *phiN1;
		*phiN1 = 0.0;
	}
}


/////////////////////////////////
//                             //
//    1D ELLIPTICAL SOLVER     //
//                             //
/////////////////////////////////



EllipticSolver1D::EllipticSolver1D(const std::string& name,MetricSpace *m,Task *tsk) : EllipticSolver(name,m,tsk)
{
	if (space->Dimensionality()!=1)
		throw tw::FatalError("EllipticSolver1D cannot be used in multi-dimensions.");
	globalIntegrator = NULL;
	if (task->globalCells[1]>1)
		globalIntegrator = new GlobalIntegrator<tw::Float>(&task->strip[1],space->Dim(3)*space->Dim(2),space->Dim(1));
	if (task->globalCells[2]>1)
		globalIntegrator = new GlobalIntegrator<tw::Float>(&task->strip[2],space->Dim(1)*space->Dim(3),space->Dim(2));
	if (task->globalCells[3]>1)
		globalIntegrator = new GlobalIntegrator<tw::Float>(&task->strip[3],space->Dim(1)*space->Dim(2),space->Dim(3));
	if (globalIntegrator==NULL)
		throw tw::FatalError("EllipticSolver1D could not create global integrator.");
}

EllipticSolver1D::~EllipticSolver1D()
{
	delete globalIntegrator;
}

void EllipticSolver1D::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	// solve div(coeff*grad(phi)) = mul*source
	// requires 1D grid

	tw::grid::axis axis;
	tw::Int s,sDim,ax,di,dj,dk;
	std::valarray<tw::Float> D(7);
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);

	di = dj = dk = 0;
	if (task->globalCells[1]>1)
	{
		axis = tw::grid::x;
		sDim = xDim;
		di = 1;
	}
	if (task->globalCells[2]>1)
	{
		axis = tw::grid::y;
		sDim = yDim;
		dj = 1;
	}
	if (task->globalCells[3]>1)
	{
		axis = tw::grid::z;
		sDim = zDim;
		dk = 1;
	}
	ax = tw::grid::naxis(axis);
	tw::strip strip(ax,*space,0,0,0);

	std::valarray<tw::Float> T1(sDim),T2(sDim),T3(sDim),src(sDim),ans(sDim);

	for (s=1;s<=sDim;s++)
	{
		FormOperatorStencil(D,1-di+di*s,1-dj+dj*s,1-dk+dk*s);
		T1[s-1] = D[1]+D[3]+D[5]; // ignorable contributions go away as required
		T2[s-1] = D[0];
		T3[s-1] = D[2]+D[4]+D[6];
		src[s-1] = mul*source(s*di,s*dj,s*dk);
	}
	if (task->n0[ax]==MPI_PROC_NULL)
		phi.AdjustTridiagonalForBoundaries(axis,tw::grid::low,T1,T2,T3,src,phi(strip,space->LFG(ax)));
	if (task->n1[ax]==MPI_PROC_NULL)
		phi.AdjustTridiagonalForBoundaries(axis,tw::grid::high,T1,T2,T3,src,phi(strip,space->UFG(ax)));
	TriDiagonal<tw::Float,tw::Float>(ans,src,T1,T2,T3);
	for (s=1;s<=sDim;s++)
		phi(strip,s) = ans[s-1];
	globalIntegrator->SetMatrix(0,T1,T2,T3);
	globalIntegrator->SetData(0,&phi(0,0,0),1);

	globalIntegrator->Parallelize();

	phi.CopyFromNeighbors();
	phi.ApplyBoundaryCondition(false);
}



/////////////////////////////////
//                             //
// ITERATIVE ELLIPTICAL SOLVER //
//                             //
/////////////////////////////////



IterativePoissonSolver::IterativePoissonSolver(const std::string& name,MetricSpace *m,Task *tsk) : EllipticSolver(name,m,tsk)
{
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);
	mask1 = new char[xDim*yDim*zDim];
	mask2 = new char[xDim*yDim*zDim];
	maxIterations = 1000;
	tolerance = 1e-8;
	const tw::Int SOR1 = tsk->globalCells[1] - (tsk->globalCells[1]==1 ? 1 : 0);
	const tw::Int SOR2 = tsk->globalCells[2] - (tsk->globalCells[2]==1 ? 1 : 0);
	const tw::Int SOR3 = tsk->globalCells[3] - (tsk->globalCells[3]==1 ? 1 : 0);
	overrelaxation = 2.0 - 10.0/(SOR1 + SOR2 + SOR3);
	minimumNorm = tw::small_pos;
	iterationsPerformed = 0;
	normSource = 0.0;
	normResidualAchieved = 0.0;
	InitializeCLProgram("elliptic.cl");

	bool redSquare = true;
	for (tw::Int k=1;k<=zDim;k++)
	{
		for (tw::Int j=1;j<=yDim;j++)
		{
			for (tw::Int i=1;i<=xDim;i++)
			{
				const tw::Int n = (i-1) + (j-1)*xDim + (k-1)*xDim*yDim;
				mask1[n] = redSquare ? 1 : 0;
				mask2[n] = redSquare ? 0 : 1;
				if (xDim>1)
					redSquare = !redSquare;
			}
			if (yDim>1)
				redSquare = !redSquare;
		}
		if (zDim>1)
			redSquare = !redSquare;
	}
	directives.Add("tolerance",new tw::input::Float(&tolerance));
	directives.Add("overrelaxation",new tw::input::Float(&overrelaxation),false);
}

IterativePoissonSolver::~IterativePoissonSolver()
{
	delete mask1;
	delete mask2;
}

void IterativePoissonSolver::FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential)
{
	EllipticSolver::FixPotential(phi,theRegion,thePotential);
	#pragma omp parallel
	{
		tw::Int i,j,k,n;
		for (auto cell : InteriorCellRange(*space))
		{
			cell.Decode(&i,&j,&k);
			n = (i-1) + (j-1)*space->Dim(1) + (k-1)*space->Dim(1)*space->Dim(2);
			if (theRegion->Inside(space->Pos(cell),*space))
			{
				mask1[n] = 0;
				mask2[n] = 0;
			}
		}
	}
}

#ifdef USE_OPENCL

void IterativePoissonSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	// solve div(coeff*grad(phi)) = mul*source

	cl_int err;
	tw::Int iter,i,j,k;
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);

	tw::Float normResidual;
	ScalarField residual;
	residual.Initialize(*space,task);
	residual.InitializeComputeBuffer();

	normSource = 0.0;
	for (k=1;k<=zDim;k++)
		for (j=1;j<=yDim;j++)
			for (i=1;i<=xDim;i++)
				normSource += fabs(mul*source(i,j,k));
	task->strip[0].AllSum(&normSource,&normSource,sizeof(tw::Float),0);
	normSource += minimumNorm;

	cl_kernel SORIterationKernel = clCreateKernel(program,coeff==NULL ? "SORIterationPoisson" : "SORIterationGeneral",&err);

	cl_mem mask1_buffer = clCreateBuffer(task->context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(char)*xDim*yDim*zDim,mask1,&err);
	cl_mem mask2_buffer = clCreateBuffer(task->context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(char)*xDim*yDim*zDim,mask2,&err);

	clSetKernelArg(SORIterationKernel,0,sizeof(cl_mem),&phi.computeBuffer);
	clSetKernelArg(SORIterationKernel,1,sizeof(cl_mem),&source.computeBuffer);
	clSetKernelArg(SORIterationKernel,2,sizeof(cl_mem),&residual.computeBuffer);
	clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask1_buffer);
	clSetKernelArg(SORIterationKernel,4,sizeof(mul),&mul);
	clSetKernelArg(SORIterationKernel,5,sizeof(overrelaxation),&overrelaxation);
	clSetKernelArg(SORIterationKernel,6,sizeof(cl_mem),&space->metricsBuffer);
	if (coeff!=NULL) clSetKernelArg(SORIterationKernel,7,sizeof(cl_mem),&coeff->computeBuffer);

	for (iter=0;iter<maxIterations;iter++)
	{
		// Update the interior (red squares)
		clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask1_buffer);
		space->LocalUpdateProtocol(SORIterationKernel,task->commandQueue);

		phi.UpdateGhostCellsInComputeBuffer();

		// Update the interior (black squares)
		clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask2_buffer);
		space->LocalUpdateProtocol(SORIterationKernel,task->commandQueue);

		phi.UpdateGhostCellsInComputeBuffer();

		// Compute norm of the residual

		normResidual = residual.DestructiveNorm1ComputeBuffer();
		task->strip[0].AllSum(&normResidual,&normResidual,sizeof(tw::Float),0);

		if (normResidual <= tolerance*normSource)
			break;
	}
	phi.ReceiveFromComputeBuffer();
	normResidualAchieved = normResidual;
	iterationsPerformed = iter;

	clReleaseMemObject(mask1_buffer);
	clReleaseMemObject(mask2_buffer);
	clReleaseKernel(SORIterationKernel);
}

#else

void IterativePoissonSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	// solve div(coeff*grad(phi)) = mul*source

	char *maskNow;
	tw::Int iter,i,j,k,ipass;
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);

	tw::Float residual,normResidual;
	tw::Float domega,rp1,rp2;
	std::valarray<tw::Float> D(7);

	normSource = 0.0;
	for (k=1;k<=zDim;k++)
		for (j=1;j<=yDim;j++)
			for (i=1;i<=xDim;i++)
				normSource += fabs(mul*source(i,j,k));
	task->strip[0].AllSum(&normSource,&normSource,sizeof(tw::Float),0);
	normSource += minimumNorm;

	for (iter=0;iter<maxIterations;iter++)
	{
		normResidual = 0.0;
		for (ipass=1;ipass<=2;ipass++)
		{
			maskNow = ipass==1 ? mask1 : mask2;

			for (k=1;k<=zDim;k++)
				for (j=1;j<=yDim;j++)
					for (i=1;i<=xDim;i++)
					{
						if (maskNow[(i-1) + (j-1)*xDim + (k-1)*xDim*yDim]==1)
						{
							FormOperatorStencil(D,i,j,k);
							residual = D[1]*phi(i-1,j,k) + D[2]*phi(i+1,j,k) + D[3]*phi(i,j-1,k) + D[4]*phi(i,j+1,k) + D[5]*phi(i,j,k-1) + D[6]*phi(i,j,k+1) + D[0]*phi(i,j,k) - mul*source(i,j,k);
							phi(i,j,k) -= overrelaxation*residual/D[0];
							normResidual += fabs(residual);
						}
					}

			phi.CopyFromNeighbors();
			phi.ApplyBoundaryCondition(false);
		}

		task->strip[0].AllSum(&normResidual,&normResidual,sizeof(tw::Float),0);
		if (normResidual <= tolerance*normSource)
			break;
	}
	normResidualAchieved = normResidual;
	iterationsPerformed = iter;
}

#endif

void IterativePoissonSolver::StatusMessage(std::ostream *theStream)
{
	*theStream << "Elliptic Solver Status:" << std::endl;
	*theStream << "   Iterations = " << iterationsPerformed << std::endl;
	*theStream << "   Overrelaxation = " << overrelaxation << std::endl;
	*theStream << "   Norm[source] = " << normSource << std::endl;
	*theStream << "   Norm[residual] = " << normResidualAchieved << std::endl;
}


///////////////////////////////
//                           //
// WAVE STYLE POISSON SOLVER //
//                           //
///////////////////////////////


PoissonSolver::PoissonSolver(const std::string& name,MetricSpace *m,Task *tsk) : EllipticSolver(name,m,tsk)
{
	globalIntegrator = NULL;
	z0 = tw::bc::fld::dirichletWall;
	z1 = tw::bc::fld::neumannWall;
	x0 = x1 = y0 = y1 = tw::bc::fld::periodic;
	if (!space->TransversePowersOfTwo())
		throw tw::FatalError("FACR elliptical solver requires all transverse dimensions be powers of two.");
	globalIntegrator = new GlobalIntegrator<tw::Float>(&task->strip[3],space->Dim(1)*space->Dim(2),space->Dim(3));
}

PoissonSolver::~PoissonSolver()
{
	delete globalIntegrator;
}

void PoissonSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	tw::Int i,j,k;
	const tw::Int xDim = phi.Dim(1);
	const tw::Int yDim = phi.Dim(2);
	const tw::Int zDim = phi.Dim(3);

	// Transform to frequency space in the transverse direction

	// Use outer ghost cells of the source to transform the boundary data for free
	if (task->n0[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			source(strip,space->LFG(3)) = phi(strip,space->LFG(3));
	if (task->n1[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			source(strip,space->UFG(3)) = phi(strip,space->UFG(3));

	switch (x0)
	{
  	case tw::bc::fld::none:
  	case tw::bc::fld::natural:
		case tw::bc::fld::normalFluxFixed:
        	break;
		case tw::bc::fld::periodic:
			source.TransverseFFT();
			break;
		case tw::bc::fld::neumannWall:
			source.TransverseCosineTransform();
			break;
		case tw::bc::fld::dirichletWall:
			source.TransverseSineTransform();
			break;
		case tw::bc::fld::dirichletCell:
			source.TransverseSineTransform();
			break;
	}

	// Copy the transformed boundary data back into the potential
	if (task->n0[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			phi(strip,space->LFG(3)) = source(strip,space->LFG(3));
	if (task->n1[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			phi(strip,space->UFG(3)) = source(strip,space->UFG(3));

	// Solve on single node

	tw::Float temp,eigenvalue,a,b,c,phi0,phiN1;
	tw::Float dz2 = sqr( gammaBeam * dz(phi) );
	a = 1.0/dz2;
	b = -2.0/dz2;
	c = 1.0/dz2;

	if (z0==tw::bc::fld::natural || z1==tw::bc::fld::natural)
		ZeroModeGhostCellValues(&phi0,&phiN1,source,mul);

	#pragma omp parallel for private(i,j,k,eigenvalue,temp) collapse(2) schedule(static)
	for (j=1;j<=yDim;j++)
		for (i=1;i<=xDim;i++)
		{
			std::valarray<tw::Float> s(zDim),u(zDim),T1(zDim),T2(zDim),T3(zDim);
			if (x0==tw::bc::fld::periodic)
				eigenvalue = phi.CyclicEigenvalue(i,j);
			else
				eigenvalue = phi.Eigenvalue(i,j);
			for (k=1;k<=zDim;k++)
			{
				s[k-1] = mul*source(i,j,k);
				T1[k-1] = a;
				T2[k-1] = b + eigenvalue;
				T3[k-1] = c;
			}
			if (task->n0[3]==MPI_PROC_NULL)
			{
				if (z0==tw::bc::fld::natural)
				{
					if (eigenvalue==0.0)
						s[0] -= T1[0]*phi0;
					else
					{
						temp = QuadraticRoot1(T1[0],T2[0],T3[0]);
						if (temp >= 1.0)
							temp = QuadraticRoot2(T1[0],T2[0],T3[0]);
						T2[0] += T1[0]*temp;
					}
				}
				else
					phi.AdjustTridiagonalForBoundaries(tw::grid::z,tw::grid::low,T1,T2,T3,s,phi(i,j,space->LFG(3)));
			}
			if (task->n1[3]==MPI_PROC_NULL)
			{
				if (z1==tw::bc::fld::natural)
				{
					if (eigenvalue==0.0)
						s[zDim-1] -= T3[zDim-1]*phiN1;
					else
					{
						temp = QuadraticRoot1(T1[zDim-1],T2[zDim-1],T3[zDim-1]);
						if (temp >= 1.0)
							temp = QuadraticRoot2(T1[zDim-1], T2[zDim-1], T3[zDim-1]);
						T2[zDim-1] += T3[zDim-1]*temp;
					}
				}
				else
					phi.AdjustTridiagonalForBoundaries(tw::grid::z,tw::grid::high,T1,T2,T3,s,phi(i,j,space->UFG(3)));
			}
			TriDiagonal<tw::Float,tw::Float>(u,s,T1,T2,T3);
			for (k=1;k<=zDim;k++)
				phi(i,j,k) = u[k-1];

			globalIntegrator->SetMatrix( (j-1)*xDim + (i-1) ,T1,T2,T3);
			globalIntegrator->SetData( (j-1)*xDim + (i-1) ,&phi(i,j,0),phi.Stride(3));
		}

	// Take into account the other nodes

	globalIntegrator->Parallelize();

	// Open boundaries must be done in transformed space

	if (task->n0[3]==MPI_PROC_NULL && z0==tw::bc::fld::natural)
	{
		for (j=1;j<=yDim;j++)
			for (i=1;i<=xDim;i++)
			{
				if (x0==tw::bc::fld::periodic)
					eigenvalue = phi.CyclicEigenvalue(i,j);
				else
					eigenvalue = phi.Eigenvalue(i,j);
				if (eigenvalue==0.0)
					phi(i,j,0) = phi0;
				else
				{
					temp = QuadraticRoot1(a,b+eigenvalue,c);
					if (temp >= 1.0)
						temp = QuadraticRoot2(a,b+eigenvalue,c);
					phi(i,j,0) = temp*phi(i,j,1);
				}
			}
	}

	if (task->n1[3]==MPI_PROC_NULL && z1==tw::bc::fld::natural)
	{
		for (j=1;j<=yDim;j++)
			for (i=1;i<=xDim;i++)
			{
				if (x0==tw::bc::fld::periodic)
					eigenvalue = phi.CyclicEigenvalue(i,j);
				else
					eigenvalue = phi.Eigenvalue(i,j);
				if (eigenvalue==0.0)
					phi(i,j,zDim+1) = phiN1;
				else
				{
					temp = QuadraticRoot1(a,b+eigenvalue,c);
					if (temp >= 1.0)
						temp = QuadraticRoot2(a,b+eigenvalue,c);
					phi(i,j,zDim+1) = temp*phi(i,j,zDim);
				}
			}
	}

	// Transform back to real space

	switch (x0)
	{
		case tw::bc::fld::none:
		case tw::bc::fld::natural:
		case tw::bc::fld::normalFluxFixed:
			break;
		case tw::bc::fld::periodic:
			phi.InverseTransverseFFT();
			//source.InverseTransverseFFT();
			break;
		case tw::bc::fld::neumannWall:
			phi.InverseTransverseCosineTransform();
			//source.InverseTransverseCosineTransform();
			break;
		case tw::bc::fld::dirichletWall:
			phi.InverseTransverseSineTransform();
			//source.InverseTransverseSineTransform();
			break;
		case tw::bc::fld::dirichletCell:
			phi.InverseTransverseSineTransform();
			//source.InverseTransverseSineTransform();
			break;
	}

	// Global boundary conditions

	phi.ApplyBoundaryCondition(false);
}



////////////////////////////////
//                            //
// CYLINDRICAL POISSON SOLVER //
//                            //
////////////////////////////////


EigenmodePoissonSolver::EigenmodePoissonSolver(const std::string& name,MetricSpace *m,Task *tsk) : EllipticSolver(name,m,tsk)
{
	x0 = tw::bc::fld::neumannWall;
	x1 = tw::bc::fld::dirichletCell;
	y0 = tw::bc::fld::periodic;
	y1 = tw::bc::fld::periodic;
	z0 = tw::bc::fld::dirichletCell;
	z1 = tw::bc::fld::dirichletCell;

	const tw::Int rDim = space->Dim(1);
	const tw::Int zDim = space->Dim(3);

	globalIntegrator = new GlobalIntegrator<tw::Float>(&task->strip[3],rDim,zDim);
}

EigenmodePoissonSolver::~EigenmodePoissonSolver()
{
	delete globalIntegrator;
}

void EigenmodePoissonSolver::Initialize()
{
	// The following call involves message passing.
	ComputeTransformMatrices(x1,eigenvalue,hankel,inverseHankel,space,task);
}

void EigenmodePoissonSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	tw::Int i,k;
	tw::Int rDim = space->Dim(1);
	tw::Int zDim = space->Dim(3);
	tw::Float temp,dz,dz1,dz2,phi0,phiN1;
	tw::Float T1_lbc,T2_lbc,T3_lbc,T1_rbc,T2_rbc,T3_rbc;
	std::valarray<tw::Float> localEig(rDim+2);

	for (i=1;i<=rDim;i++)
		localEig[i] = eigenvalue[i-1+task->cornerCell[1]-1];

	// Matrix elements for open boundary condition

	T1_lbc = T3_lbc = 1.0/sqr(space->dX(0,3));
	T2_lbc = -2.0/sqr(space->dX(0,3)); // add the eigenvalue later
	T1_rbc = T3_rbc = 1.0/sqr(space->dX(zDim+1,3));
	T2_rbc = -2.0/sqr(space->dX(zDim+1,3)); // add the eigenvalue later

	// Use outer ghost cells of the source to transform the boundary data for free
	if (task->n0[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			source(strip,space->LFG(3)) = phi(strip,space->LFG(3));
	if (task->n1[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			source(strip,space->UFG(3)) = phi(strip,space->UFG(3));

	// Perform hankel transform

	source.Hankel(task->globalCells[1],hankel);

	// Copy the transformed boundary data back into the potential
	if (task->n0[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			phi(strip,space->LFG(3)) = source(strip,space->LFG(3));
	if (task->n1[3]==MPI_PROC_NULL)
		for (auto strip : StripRange(*space,3,strongbool::no))
			phi(strip,space->UFG(3)) = source(strip,space->UFG(3));

	// Solve on single node

	if (z0==tw::bc::fld::natural || z1==tw::bc::fld::natural)
		ZeroModeGhostCellValues(&phi0,&phiN1,source,mul);

	#pragma omp parallel for private(i,k,temp) schedule(static)
	for (i=1;i<=rDim;i++)
	{
		std::valarray<tw::Float> s(zDim),u(zDim),T1(zDim),T2(zDim),T3(zDim);
		for (k=1;k<=zDim;k++)
		{
			s[k-1] = mul*source(i,0,k);
			dz = space->dX(k,3);
			dz1 = 0.5*(space->dX(k-1,3)+dz);
			dz2 = 0.5*(space->dX(k+1,3)+dz);
			T1[k-1] = 1.0/(dz*dz1);
			T2[k-1] = -(1.0/(dz*dz1) + 1.0/(dz*dz2)) + localEig[i];
			T3[k-1] = 1.0/(dz*dz2);
		}
		if (task->n0[3]==MPI_PROC_NULL)
		{
			if (z0==tw::bc::fld::natural)
			{
				if (localEig[i]==0.0)
					s[0] -= T1[0]*phi0;
				else
				{
					temp = QuadraticRoot1(T1_lbc,T2_lbc+localEig[i],T3_lbc);
					if (temp >= 1.0)
						temp = QuadraticRoot2(T1_lbc,T2_lbc+localEig[i],T3_lbc);
					T2[0] += T1[0]*temp;
				}
			}
			else
				phi.AdjustTridiagonalForBoundaries(tw::grid::z,tw::grid::low,T1,T2,T3,s,phi(i,0,space->LFG(3)));
		}
		if (task->n1[3]==MPI_PROC_NULL)
		{
			if (z1==tw::bc::fld::natural)
			{
				if (localEig[i]==0.0)
					s[zDim-1] -= T3[zDim-1]*phiN1;
				else
				{
					temp = QuadraticRoot1(T1_rbc,T2_rbc+localEig[i],T3_rbc);
					if (temp >= 1.0)
						temp = QuadraticRoot2(T1_rbc,T2_rbc+localEig[i],T3_rbc);
					T2[zDim-1] += T3[zDim-1]*temp;
				}
			}
			else
				phi.AdjustTridiagonalForBoundaries(tw::grid::z,tw::grid::high,T1,T2,T3,s,phi(i,0,space->UFG(3)));
		}

		TriDiagonal<tw::Float,tw::Float>(u,s,T1,T2,T3);
		for (k=1;k<=zDim;k++)
			phi(i,0,k) = u[k-1];

		globalIntegrator->SetMatrix(i-1,T1,T2,T3);
		globalIntegrator->SetData(i-1,&phi(i,0,0),phi.Stride(3));
	}

	// Take into account the other nodes

	globalIntegrator->Parallelize();

	// Open boundaries have to be done in transformed space

	if (task->n0[3]==MPI_PROC_NULL && z0==tw::bc::fld::natural)
	{
		for (i=1;i<=rDim;i++)
		{
			if (localEig[i]==0.0)
				phi(i,0,0) = phi0;
			else
			{
				temp = QuadraticRoot1(T1_lbc,T2_lbc+localEig[i],T3_lbc);
				if (temp >= 1.0)
					temp = QuadraticRoot2(T1_lbc,T2_lbc+localEig[i],T3_lbc);
				phi(i,0,0) = temp*phi(i,0,1);
			}
		}
	}

	if (task->n1[3]==MPI_PROC_NULL && z1==tw::bc::fld::natural)
	{
		for (i=1;i<=rDim;i++)
		{
			if (localEig[i]==0.0)
				phi(i,0,zDim+1) = phiN1;
			else
			{
				temp = QuadraticRoot1(T1_rbc,T2_rbc+localEig[i],T3_rbc);
				if (temp >= 1.0)
					temp = QuadraticRoot2(T1_rbc,T2_rbc+localEig[i],T3_rbc);
				phi(i,0,zDim+1) = temp*phi(i,0,zDim);
			}
		}
	}

	// back to real space

	phi.InverseHankel(task->globalCells[1],inverseHankel);
	//source.InverseHankel(task->globalCells[1],inverseHankel);

	// Global boundary conditions

	phi.ApplyBoundaryCondition(false);
}



/////////////////////////////////
//                             //
// MULTIGRID ELLIPTICAL SOLVER //
//                             //
/////////////////////////////////



MultiGridSolver::MultiGridSolver(const std::string& name,MetricSpace *m,Task *tsk) : EllipticSolver(name,m,tsk)
{

	// nx, ny, nz
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);

	if(!space->PowersOfTwo())
		throw tw::FatalError("MultiGrid solver requires all dimensions be in powers of 2.");
	if(space->Dimensionality()!=2)
		throw tw::FatalError("MultiGrid solver can only be used in in 2 dimensions.");


	maxIterations = 1000; //30 from the driver
	GSIterations = 20;
	GS_iter = 20;
	tolerance = 1e-8;

	nx = xDim==1 ? yDim : xDim;
	ny = yDim==1 ? zDim : yDim;
	// nz = space->Dim(3);

	// find dx, dy, dz (cell size)
	dx = space->dX(nx / 2, 1);
	dy = space->dX(ny / 2, 2);

	// const tw::Float dx = space->dX(xDim / 2, 1);
	// const tw::Float dy = space->dX(yDim / 2, 2);
	// const tw::Float dz = space->dX(zDim / 2, 3);
	//NOTE: xDim * dx gives the actual "size" of the grid

	// SOR1-3 is the dimensions of x, y, z (if they are dim = 1 SOR is 0)
	// const tw::Int SOR1 = tsk->globalCells[1] - (tsk->globalCells[1]==1 ? 1 : 0);
	// const tw::Int SOR2 = tsk->globalCells[2] - (tsk->globalCells[2]==1 ? 1 : 0);
	// const tw::Int SOR3 = tsk->globalCells[3] - (tsk->globalCells[3]==1 ? 1 : 0);
	// overrelaxation = 2.0 - 10.0/(SOR1 + SOR2 + SOR3);

	// determine the number of MG levels along "x" and "y"
	nx_levels=int(log2(nx+0.1))-1;                            // number of levels along "x"
  	ny_levels=int(log2(ny+0.1))-1;                            // number of levels along "y"
	// const tw::Float nz_levels=int(log2(zDim+0.1))-1;                            // number of levels along "z"

	// determine the number of MG levels for the whole thing (the min of nx ny nz levels)
	// levels=nz_levels;if (nx_levels < levels) levels=nx_levels; if (ny_levels < levels) levels=ny_levels;
	levels=ny_levels;if (nx_levels < ny_levels) levels=nx_levels;

	a  = new double *[ny+1];for (i=0;i<=ny;i++) a[i]  = new double [nx+1];
	b  = new double *[ny+1];for (i=0;i<=ny;i++) b[i]  = new double [nx+1];
	c  = new double *[ny+1];for (i=0;i<=ny;i++) c[i]  = new double [nx+1];
	d  = new double *[ny+1];for (i=0;i<=ny;i++) d[i]  = new double [nx+1];
	e  = new double *[ny+1];for (i=0;i<=ny;i++) e[i]  = new double [nx+1];
	r  = new double *[ny+1];for (i=0;i<=ny;i++) r[i]  = new double [nx+1];
	u_tmp = new double *[ny+1];for (i=0;i<=ny;i++) u_tmp[i] = new double [nx+1];
	nl = new int [ny_levels+1];ml = new int [nx_levels+1];
	ul = new double **[levels+1];
	for (l=0;l<=levels;l++) {                                 // begin loop over levels
		l1=pow(2,ny_levels-levels+l+1)+1;                       // number of grid points along "y"
		l2=pow(2,nx_levels-levels+l+1)+1;                       // number of grid points along "x"
		ul[l]= new double *[l1];                                // reserve memory for "ul"
		for (i=0;i<=l1;i++) ul[l][i] = new double [l2];         // reserve memory for "ul"
	}                                                         // end loop over levels
	relat_error = new double [maxIterations+1];                    // reserve memory for "relat_error"
	resid_error = new double [maxIterations+1];                    // reserve memory for "resid_error"
	
	for (l=0;l<=levels;l++) {
		nl[l]=pow(2,ny_levels-levels+l+1);                      // set # points at level "l"
		ml[l]=pow(2,nx_levels-levels+l+1);                      // set # points at level "l"
	}

	directives.Add("tolerance",new tw::input::Float(&tolerance));
	// directives.Add("overrelaxation",new tw::input::Float(&overrelaxation),false);
	coefficients.Initialize(*space,task);
}

MultiGridSolver::~MultiGridSolver()
{
	delete mask1;
	delete mask2;
}

void MultiGridSolver::FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential)
{
	EllipticSolver::FixPotential(phi,theRegion,thePotential);
	#pragma omp parallel
	{
		tw::Int i,j,k,n;
		for (auto cell : InteriorCellRange(*space))
		{
			cell.Decode(&i,&j,&k);
			n = (i-1) + (j-1)*space->Dim(1) + (k-1)*space->Dim(1)*space->Dim(2);
			if (theRegion->Inside(space->Pos(cell),*space))
			{
				mask1[n] = 0;
				mask2[n] = 0;
			}
		}
	}
}

#ifdef USE_OPENCL

void MultiGridSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	// solve div(coeff*grad(phi)) = mul*source

	cl_int err;
	tw::Int iter,i,j,k;
	const tw::Int xDim = space->Dim(1);
	const tw::Int yDim = space->Dim(2);
	const tw::Int zDim = space->Dim(3);

	tw::Float normResidual;
	ScalarField residual;
	residual.Initialize(*space,task);
	residual.InitializeComputeBuffer();

	normSource = 0.0;
	for (k=1;k<=zDim;k++)
		for (j=1;j<=yDim;j++)
			for (i=1;i<=xDim;i++)
				normSource += fabs(mul*source(i,j,k));
	task->strip[0].AllSum(&normSource,&normSource,sizeof(tw::Float),0);
	normSource += minimumNorm;

	cl_kernel SORIterationKernel = clCreateKernel(program,coeff==NULL ? "SORIterationPoisson" : "SORIterationGeneral",&err);

	cl_mem mask1_buffer = clCreateBuffer(task->context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(char)*xDim*yDim*zDim,mask1,&err);
	cl_mem mask2_buffer = clCreateBuffer(task->context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(char)*xDim*yDim*zDim,mask2,&err);

	clSetKernelArg(SORIterationKernel,0,sizeof(cl_mem),&phi.computeBuffer);
	clSetKernelArg(SORIterationKernel,1,sizeof(cl_mem),&source.computeBuffer);
	clSetKernelArg(SORIterationKernel,2,sizeof(cl_mem),&residual.computeBuffer);
	clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask1_buffer);
	clSetKernelArg(SORIterationKernel,4,sizeof(mul),&mul);
	clSetKernelArg(SORIterationKernel,5,sizeof(overrelaxation),&overrelaxation);
	clSetKernelArg(SORIterationKernel,6,sizeof(cl_mem),&space->metricsBuffer);
	if (coeff!=NULL) clSetKernelArg(SORIterationKernel,7,sizeof(cl_mem),&coeff->computeBuffer);

	for (iter=0;iter<maxIterations;iter++)
	{
		// Update the interior (red squares)
		clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask1_buffer);
		space->LocalUpdateProtocol(SORIterationKernel,task->commandQueue);

		phi.UpdateGhostCellsInComputeBuffer();

		// Update the interior (black squares)
		clSetKernelArg(SORIterationKernel,3,sizeof(cl_mem),&mask2_buffer);
		space->LocalUpdateProtocol(SORIterationKernel,task->commandQueue);

		phi.UpdateGhostCellsInComputeBuffer();

		// Compute norm of the residual

		normResidual = residual.DestructiveNorm1ComputeBuffer();
		task->strip[0].AllSum(&normResidual,&normResidual,sizeof(tw::Float),0);

		if (normResidual <= tolerance*normSource)
			break;
	}
	phi.ReceiveFromComputeBuffer();
	normResidualAchieved = normResidual;
	iterationsPerformed = iter;

	clReleaseMemObject(mask1_buffer);
	clReleaseMemObject(mask2_buffer);
	clReleaseKernel(SORIterationKernel);
}

#else

void MultiGridSolver::Solve(ScalarField& phi,ScalarField& source,tw::Float mul)
{
	// solve div(coeff*grad(phi)) = mul*source

	tw::Float residual,normResidual;
	tw::Float domega,rp1,rp2;
	std::valarray<tw::Float> D(7);
	// copy input array data into local class data
	for (i=0;i<=ny;i++) {
		for (j=0;j<=nx;j++) {
		// this->u[i][j]=phi(i, 1, j);
		// this->dd[i][j]= 8.854187817e-12;//(*coeff)(i, 1, j)
		coefficients(i, 1, j)=8.854187817e-12;
		// if(iter == 0) {phi(i, 1, j)=200;}
		// this->f[i][j]=source(i, 1, j);                               // copy f1->f
		}
	}
	// // from solve method
	for (iter=1;iter<=maxIterations;iter++) {
		// std::string name = "Hello, " +  std::to_string(iter);
		// throw tw::FatalError(name);
		for (i=0;i<=ny;i++) {for (j=0;j<=nx;j++) u_tmp[i][j]=phi(i, 1, j);}
		for (l=levels;l>=0;l--) {
		if (l < levels) {
			for (i=0;i<=nl[l];i++) for (j=0;j<=ml[l];j++) phi(i, 1, j)=0.0;
		}     

		// smoothing(l)
		this->Smoothing(phi, source, l);
		//end of smoothing();
	
		//keep_ul();
		for (i=0;i<=nl[l];i++) {for (j=0;j<=ml[l];j++) ul[l][i][j]=phi(i, 1, j);}

		//restriction of defect()
		if (l > 0){
			i=0;for (j=0;j<=ml[l-1];j++)         source(i, 1, j)=r[2*i][2*j]; // i=0
			i=nl[l-1];for (j=0;j<=ml[l-1];j++)   source(i, 1, j)=r[2*i][2*j]; // i=nl[l-1]
			j=0;for (i=1;i<=nl[l-1]-1;i++)       source(i, 1, j)=r[2*i][2*j]; // j=0
			j=ml[l-1];for (i=1;i<=nl[l-1]-1;i++) source(i, 1, j)=r[2*i][2*j]; // j=ml[l-1]
			// grid points in the interior
			for (i=1;i<=nl[l-1]-1;i++) {                              // begin loop over "i"
				for (j=1;j<=ml[l-1]-1;j++) {                            // begin loop over "j"
				source(i, 1, j)=r[2*i-1][2*j-1]/16.0+r[2*i-1][2*j]/8.0+r[2*i-1][2*j+1]/16.0+
						r[2*i+0][2*j-1]/8.00+r[2*i+0][2*j]/4.0+r[2*i+0][2*j+1]/8.00+
						r[2*i+1][2*j-1]/16.0+r[2*i+1][2*j]/8.0+r[2*i+1][2*j+1]/16.0;
				}                                                       // end loop over "j"
			} 
		} 
		//end of restriction of defect();
		
		}                                                       // end "winding"
		for (l=1;l<=levels;l++) {                               // begin "unwinding"
			//prolongation
			for (i=0;i<=nl[l-1]-1;i++) {                              // begin loop over "i"
				for (j=0;j<=ml[l-1]-1;j++) {                            // begin loop over "j"
				ul[l][2*i][2*j]-=phi(i, 1, j);                             // even "i" & even "j"
				ul[l][2*i+1][2*j]-=(phi(i, 1, j)+phi(i+1, 1, j))/2.0;           // odd  "i" & even "j"
				ul[l][2*i][2*j+1]-=(phi(i, 1, j)+phi(i, 1, j+1))/2.0;           // even "i" & odd  "j"
				ul[l][2*i+1][2*j+1]-=(phi(i, 1, j)+phi(i, 1, j+1)+phi(i+1, 1, j)+phi(i+1, 1, j+1))/4.0;// odd "i" & odd "j"
				}                                                       // end loop over "j"
				j=ml[l-1];ul[l][2*i][2*j]-=phi(i, 1, j);                     // even "i" & even "j"
				ul[l][2*i+1][2*j]-=(phi(i, 1, j)+phi(i+1, 1, j))/2.0;             // odd  "i" & even "j"
			}                                                         // end loop over "i"
			i=nl[l-1];                                                // last row with i=nl[l-1]
			for (j=0;j<=ml[l-1]-1;j++) {                              // begin loop over "j"
				ul[l][2*i][2*j]-=phi(i, 1, j);                               // even "i" & even "j"
				ul[l][2*i][2*j+1]-=(phi(i, 1, j)+phi(i, 1, j+1))/2.0;             // even "i" & odd  "j"
			}                                                         // end loop over "j"
			j=ml[l-1];ul[l][2*i][2*j]-=phi(i, 1, j);                       // top right corner
			// put solution ul[l][i][j] at level "l" back into u[i][j], i.e. ul[l][i][j] -> u[i][j]
			for (i=0;i<=nl[l];i++) {for (j=0;j<=ml[l];j++) phi(i, 1, j)=ul[l][i][j];}
			//end of prolongation
		}                                                       // end "unwinding"
		relat_error[iter]=relative_error;                       // relative error
		resid_error[iter]=residual_error;                       // error of residual
		
		if (residual_error < tol) break;
	}

	normResidualAchieved = normResidual;
	iterationsPerformed = iter;
	std::string name = "Hello, " +  std::to_string(iterationsPerformed);
	throw tw::FatalError(name);
}

#endif

void MultiGridSolver::Smoothing(ScalarField& phi,ScalarField& source,tw::Int l)
{
	tw::Float sum1,sum2,sum3,sum4,dX2,dY2;                       // local variables
	tw::Int it,is,ic,in,jw,jc,je,step;                          // local variables
	step=pow(2,levels-l);                                     // step depending on level "l"
	dX2=2.0*sqr(dx*step);                                     // 2*(step*dx)^2
	dY2=2.0*sqr(dy*step);                                     // 2*(step*dy)^2
	// set up coefficients "a", "b", "c" and "d" on interior points
	for (i=1;i<=nl[l]-1;i++) {                                // begin loop over "i"
		for (j=1;j<=ml[l]-1;j++) {                              // begin loop over "j"
		is=(i-1)*step;ic=i*step;in=(i+1)*step;                // i-1,i,i+1
		jw=(j-1)*step;jc=j*step;je=(j+1)*step;                // j-1,j,j+1
		a[i][j]=(coefficients(ic, 1, jw)+coefficients(ic, 1, jc))/dX2;                    // set a[i][j]
		c[i][j]=(coefficients(ic, 1, je)+coefficients(ic, 1, jc))/dX2;                    // set c[i][j]
		d[i][j]=(coefficients(in, 1, jc)+coefficients(ic, 1, jc))/dY2;                    // set d[i][j]
		e[i][j]=(coefficients(is, 1, jc)+coefficients(ic, 1, jc))/dY2;                    // set e[i][j]
		b[i][j]=-a[i][j]-c[i][j]-d[i][j]-e[i][j];             // set b[i][j]
		}                                                       // end loop over "j"
	}                                                         // end loop over "i"
	// calculate residual "r" and solution "u" on interior points
	for (it=1;it<=GS_iter;it++) {                       // begin iterations
		for (i=1;i<=nl[l]-1;i++) {                              // begin loop over "i"
		for (j=1;j<=ml[l]-1;j++) {                            // begin loop over "j"
			r[i][j]=(a[i][j]*phi(i, 1, j-1)+b[i][j]*phi(i, 1, j)+c[i][j]*phi(i, 1, j+1)+d[i][j]*phi(i+1, 1, j)+e[i][j]*phi(i-1, 1, j)-source(i, 1, j));
		}                                                     // end loop over "j"
		}                                                       // end loop over "i"
		for (i=1;i<=nl[l]-1;i++) {                              // begin loop over "i"
		for (j=1;j<=ml[l]-1;j++) {                            // begin loop over "j"
			phi(i, 1, j)-=r[i][j]/b[i][j];                           // u[i][j]=...
		}                                                     // end loop over "j"
		}                                                       // end loop over "i"
	}                                                         // end iterations
	// if on original grid (highest level), calculate relative errors
	
	if (l == levels) {                                        // begin error of the residual
		sum1=0.0;sum2=0.0;sum3=0.0;sum4=0.0;                    // set sum to 0
		for (i=1;i<=nl[l]-1;i++) {                              // begin loop over "i"
		for (j=1;j<=ml[l]-1;j++) {                            // begin loop over "j"
			sum1+=sqr(r[i][j]);sum2+=sqr(source(i, 1, j));              // accumulate in sum1 and sum2
			sum3+=sqr(phi(i, 1, j)-u_tmp[i][j]);sum4+=sqr(phi(i, 1, j));  // accumulate in sum3 and sum4
		}                                                     // end loop over "j"
		}                                                       // end loop over "i"
		residual_error=sqrt(sum1/sum2);                         // relative error of the residual
		relative_error=sqrt(sum3/sum4);                         // relative error between two iter.
	} 
}

void MultiGridSolver::StatusMessage(std::ostream *theStream)
{
	*theStream << "Elliptic Solver Status:" << std::endl;
	*theStream << "   Iterations = " << iterationsPerformed << std::endl;
	// *theStream << "   Overrelaxation = " << overrelaxation << std::endl;
	*theStream << "   Norm[source] = " << normSource << std::endl;
	*theStream << "   Norm[residual] = " << normResidualAchieved << std::endl;
}