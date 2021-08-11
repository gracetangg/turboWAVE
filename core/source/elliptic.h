// IterativePoissonSolver and EllipticSolver1D handle equations of the form div(coeff*grad(phi)) = mul*source
// PoissonSolver and EigenmodePoissonSolver only handle div(phi) = mul*source
// For PoissonSolver and EigenmodePoissonSolver, only z-boundary conditions are programmable

// Boundary Conditions:
// These tools expect boundary values to be stored in the outer ghost cells
// E.g., if phi(i,j,-1) = V0 then:
// BC Type          Explicit BC
// -------------    ------------
// dirichletCell    phi(i,j,0) = V0
// neumannWall      phi(i,j,1) - phi(i,j,0) = 0
// dirichletWall    phi(i,j,0)/2 + phi(i,j,1)/2 = V0

// N.b. the BC handling assumes at least 2 ghost cell layers in the field

struct EllipticSolver:BoundedTool
{
	ScalarField *coeff;
	tw::Float gammaBeam;

	EllipticSolver(const std::string& name,MetricSpace *m,Task *tsk);
	virtual void SetCoefficients(ScalarField *coefficients);
	virtual void FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential);
	virtual void ZeroModeGhostCellValues(tw::Float *phi0,tw::Float *phiN1,ScalarField& rho,tw::Float mul);
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul) = 0;
	void FormOperatorStencil(std::valarray<tw::Float>& D,tw::Int i,tw::Int j,tw::Int k);
};

struct IterativePoissonSolver:EllipticSolver
{
	char *mask1,*mask2;
	tw::Int iterationsPerformed;
	tw::Float normResidualAchieved,normSource;
	tw::Int maxIterations;
	tw::Float tolerance,overrelaxation,minimumNorm;

	IterativePoissonSolver(const std::string& name,MetricSpace *m,Task *tsk);
	virtual ~IterativePoissonSolver();
	virtual void FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential);
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul);
	virtual void StatusMessage(std::ostream *theStream);
};

struct EllipticSolver1D:EllipticSolver
{
	GlobalIntegrator<tw::Float> *globalIntegrator;

	EllipticSolver1D(const std::string& name,MetricSpace *m,Task *tsk);
	virtual ~EllipticSolver1D();
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul);
};

struct PoissonSolver:EllipticSolver
{
	GlobalIntegrator<tw::Float> *globalIntegrator;

	PoissonSolver(const std::string& name,MetricSpace *m,Task *tsk);
	virtual ~PoissonSolver();
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul);
};

struct EigenmodePoissonSolver:EllipticSolver
{
	std::valarray<tw::Float> eigenvalue,hankel,inverseHankel;
	GlobalIntegrator<tw::Float> *globalIntegrator;

	EigenmodePoissonSolver(const std::string& name,MetricSpace *m,Task *tsk);
	virtual ~EigenmodePoissonSolver();
	virtual void Initialize();
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul);
};

struct MultiGridSolver:EllipticSolver
{
	char *mask1,*mask2;
	tw::Float  ***ul;                                           // numerical solution of the PDE
    tw::Float  **a;                                             // coefficient for u[i][j-1]
    tw::Float  **b;                                             // coefficient for u[i][j]
    tw::Float  **c;                                             // coefficient for u[i][j+1]
    tw::Float  **d;                                             // coefficient for u[i+1][j]
    tw::Float  **e;
  
	ScalarField coefficients;                                    // right hand side of the PDE
    tw::Float  **r;                                             // residual of the PDE
    tw::Float  **u_tmp;                                         // solution of the PDE prev. iter.
    tw::Float  *relat_error;                                    // relative error between iterations
    tw::Float  *resid_error;                                    // relative error of the residual
    tw::Int *ml;                                             // number of grid points at level "l"
    tw::Int *nl;                                             // number of grid points at level "l"
    tw::Int nx_levels;                                       // number of levels along "x"
    tw::Int ny_levels;                                       // number of levels along "y"
    tw::Int l;                                               // current level: l=0...level
    tw::Int nx;                                              // number of grid points along "x"
    tw::Int ny;                                              // number of grid points along "y"
    tw::Int nz;
	tw::Int i,j;                                             // indices: i=0...ny, j=0...nx
    tw::Int iter;                                            // global iteration counter
    tw::Int levels;                                          // number of MG levels
    tw::Int GS_iter;                                         // number of Gauss-Seidel iter
    tw::Int max_iter;                                        // maximum number of MG iterations
    tw::Float  dx,dy;                                           // steps along "x" and "y"
	tw::Int l1, l2;

	//public from driver
    tw::Int nmb_iter;                                           // actual number of MGM iterations
    tw::Float  tol;                                             // input acceptable tolerance error
    tw::Float  residual_error;                                  // error of the residual
    tw::Float  relative_error;                                  // error between two successive iter.

	//originally from iterative solver
	tw::Int iterationsPerformed;
	tw::Float normResidualAchieved,normSource;
	tw::Int maxIterations;
	tw::Int GSIterations;
	tw::Float tolerance,overrelaxation,minimumNorm;

	MultiGridSolver(const std::string& name,MetricSpace *m,Task *tsk);
	virtual ~MultiGridSolver();
	virtual void FixPotential(ScalarField& phi,Region* theRegion,const tw::Float& thePotential);
	virtual void Solve(ScalarField& phi,ScalarField& source,tw::Float mul);
	virtual void StatusMessage(std::ostream *theStream);
	virtual void Smoothing(ScalarField& phi,ScalarField& source,tw::Int l);
};
