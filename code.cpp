#define _USE_MATH_DEFINES
#include <iostream>
#include <utility>  

#include <iomanip> 
#include <stdexcept>       

#include <fstream>
#include <chrono> 
#include <string>
#include <vector>
#include <cmath>
#include <algorithm> 
#include <Eigen/Dense>       

using namespace std;
using namespace Eigen;



vector<double> multiplyMatrixVector(const vector<vector<double>>& matrix, const vector<double>& vec) {

    vector<double> result(matrix.size(), 0.0);

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

class SetGrid {
public: 
    size_t size;
    vector<double> u; //grid points from 0 to 1
    vector<double>  z; //chebyshev points from 1 to -1
    vector<vector<double>> D;  //chebyshev_diff_matrix
    vector<vector<double>> D2;
    
    SetGrid(size_t n) : size(n), z(n), u(n), D(n, vector<double>(n)){
        
        //set u_grid
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            z[i] = cos((M_PI * i) / (n - 1));
        }

        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            u[i] = z[i] * 0.5 + 0.5;
        }
    
        std::reverse(u.begin(), u.end());
    

        //chebyshev_diff_matrix
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                if (i == static_cast<size_t>(n - 1) && j == static_cast<size_t>(n - 1)) {
                    D[i][j] = -((2.0 * (n - 1.0) * (n - 1) + 1.0) / 6.0);
                }
                else if (i == 0 && j == 0) {
                    D[i][j] = (2.0 * (n - 1.0) * (n - 1) + 1.0) / 6.0;
                }
                else if (i == j) {
                    D[i][j] = -z[i] / (2.0 * (1.0 - z[i] * z[i]));
                }
                else {
                    D[i][j] = (((i + j) % 2 == 0 ? 1.0 : -1.0) / (z[i] - z[j]));

                    

                    if (i == 0 || i == n - 1) D[i][j] *= 2.0;
                    if (j == 0 || j == n - 1) D[i][j] *= 1.0 / 2.0;
                }
            }
        }


        // Scale the differentiation matrix for the interval [0, 1]
        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                D[i][j] *= 2;
            }
        }

        // Reverse rows and columns to match increasing order in [0, 1]
        vector<vector<double>> Dtemp = D;

        for (size_t i = 0; i < static_cast<size_t>(n); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                D[i][j] = Dtemp[static_cast<size_t>(n - 1 - i)][static_cast<size_t>(n - 1 - j)];
            }
        }
    
        D2 = multiplyMatrixMatrix(D, D);


    }

          

private:
        
        //function to compute the product of 2 matrices
        vector<vector<double>> multiplyMatrixMatrix(const vector<vector<double>>& mat1, const vector<vector<double>>& mat2) {
            size_t rows1 = mat1.size();
            size_t cols1 = mat1[0].size();
            size_t cols2 = mat2[0].size();

            vector<vector<double>> product(rows1, vector<double>(cols2, 0.0));

            for (size_t i = 0; i < rows1; ++i) {
                for (size_t j = 0; j < cols2; ++j) {
                    for (size_t k = 0; k < cols1; ++k) {
                        product[i][j] += mat1[i][k] * mat2[k][j];
                    }
                }
            }
            return product;
        }
};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class coefficients {
public:
    vector<vector<double>> coeff_matrix;
    vector<double> source;

    coefficients(SetGrid& grid) : N(grid.size), coeff_matrix(grid.size, vector<double>(grid.size, 0.0)), source(grid.size, 0.0), 
        unit_matrix(grid.size, vector<double>(grid.size, 0.0)) {

        for (size_t i = 0; i < N; i++) {
            unit_matrix[i][i] = 1.0;
        }
    }
    
    //update the coeffs of the eom number m. m goes from 1 to 4.
    void update(size_t& m, SetGrid& grid_inst, vector<vector<vector<double>>>& data, double& ksi, double& a4){

        switch (m) {
        case 1: // Ss
            coeff_matrix[0][0] = 40.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = 2.0 * pow(grid_inst.u[i], 2) * grid_inst.D2[i][j] + 20.0 * grid_inst.u[i] * 
                        grid_inst.D[i][j] + (40.0 + pow(grid_inst.u[i], 8) * pow(4 * data[0][0][i] + grid_inst.u[i] * 
                            data[0][1][i], 2)) * unit_matrix[i][j];
                }
            }

            source[0] = 0;
            for (size_t i = 1; i < N; ++i) {
                source[i] = -pow(grid_inst.u[i], 3) * (1.0 + grid_inst.u[i] * ksi) * pow(4 * data[0][0][i] + grid_inst.u[i] * 
                    data[0][1][i], 2);
            }
            break;
        case 2: // Sdots

            source[0] = 0.5 * a4;//bdry condition

            for (size_t i = 1; i < N; ++i) {
                source[i] = 2 * pow(grid_inst.u[i], 5) * pow(data[1][0][i], 2) + data[1][0][i] * (7.0 + 11.0 * grid_inst.u[i] * ksi + 4 *
                    pow(grid_inst.u[i], 2) * pow(ksi, 2)) + grid_inst.u[i] * pow(1 + grid_inst.u[i] * ksi, 2) * data[1][1][i];
            }

            // Create the coefficient matrix 
            coeff_matrix[0][0] = 1.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = (-pow(grid_inst.u[i], 5) * data[1][0][i] - (1.0 + grid_inst.u[i] * ksi)) * grid_inst.D[i][j] -
                        2.0 * (ksi + 5.0 * pow(grid_inst.u[i], 4) * data[1][0][i] + pow(grid_inst.u[i], 5) * data[1][1][i]) * 
                        unit_matrix[i][j];
                }
            }
            break;
        case 3: // Bdots
            source[0] = 6.0 * data[0][0][0];
            for (size_t i = 1; i < N; ++i) {
                source[i] = 6.0 * data[0][0][i] * (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i] ) +
                    0.5 * 3.0 * grid_inst.u[i] * (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i]) * 
                    data[0][1][i];
            }

            // Create the coefficient matrix
           
            coeff_matrix[0][0] = -3.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = -2.0 * grid_inst.u[i] * (1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i]) *
                        grid_inst.D[i][j] - 3.0 * (1.0 + 2.0 * grid_inst.u[i] * ksi + 6.0 * pow(grid_inst.u[i], 5) * data[1][0][i] +
                            pow(grid_inst.u[i], 6) * data[1][1][i]) * unit_matrix[i][j];
                }
            }
            break;
        case 4: // As

            source[0] = a4;//bdry condition

            for (size_t i = 1; i < N; ++i) {
                source[i] = 3.0 * pow(grid_inst.u[i], 4) * data[3][0][i] * (4.0 * data[0][0][i] + grid_inst.u[i] * data[0][1][i]) *
                    pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i] , 2) -(6.0 / pow(grid_inst.u[i], 4)) * 
                    pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) * data[1][0][i] , 2) - (6.0 / pow(grid_inst.u[i], 4)) * 
                    (pow(1.0 + grid_inst.u[i] * ksi, 2) + 2.0 * pow(grid_inst.u[i], 4) * data[2][0][i] ) * (-1.0 + 4.0 * 
                        pow(grid_inst.u[i], 5) * data[1][0][i]  + pow(grid_inst.u[i], 6) * data[1][1][i]);
            }

            // Create the coefficient matrix 
            coeff_matrix[0][0] = 1.0;

            for (size_t i = 1; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    coeff_matrix[i][j] = pow(grid_inst.u[i], 2) * pow(1.0 + grid_inst.u[i] * ksi + pow(grid_inst.u[i], 5) *
                        data[1][0][i], 2) * grid_inst.D2[i][j] + 6.0 * grid_inst.u[i] * pow(1.0 + grid_inst.u[i] * ksi + 
                            pow(grid_inst.u[i], 5) * data[1][0][i] , 2) * grid_inst.D[i][j] + 6.0 * pow(1.0 + grid_inst.u[i] * ksi +
                                pow(grid_inst.u[i], 5) * data[1][0][i] , 2) * unit_matrix[i][j];
                }
            }

            break;
        }

    
    }
private:
    size_t N;
    vector<vector<double>> unit_matrix;
    
};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class solve{

public:
    coefficients coeff;

    solve(SetGrid& grid) : grid_inst(grid), coeff(grid) {}

    //solve the first num equations.
        void run(size_t num, vector<vector<vector<double>>>& data, double& ksi, double& a4) {
            for (size_t i = 1; i <= num; ++i) {
                coeff.update(i, grid_inst, data, ksi, a4);
                  

            // conversion to Eigen
            Eigen_coeff_matrix = convert_stdMatrix_To_MatrixXd(coeff.coeff_matrix);
            Eigen_source = Eigen::Map<Eigen::VectorXd>(coeff.source.data(), coeff.source.size());

            // solve
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Eigen_coeff_matrix);
            Eigen_sol = lu.solve(Eigen_source);

            // Convert vectorXd to vector
            data[i][0] = convert_vactorXd_to_stdVector(Eigen_sol);
            data[i][1] = multiplyMatrixVector(grid_inst.D, data[i][0]);
            data[i][2] = multiplyMatrixVector(grid_inst.D2, data[i][0]);
            
        }
    
        }

private: 
    SetGrid& grid_inst;
    Eigen::MatrixXd Eigen_coeff_matrix;
    Eigen::VectorXd Eigen_source;
    Eigen::VectorXd Eigen_sol;
    

    // Function to convert Matrix to a matrixXd
    Eigen::MatrixXd convert_stdMatrix_To_MatrixXd(const vector<vector<double>>& matrix_xd) {
        size_t rows = matrix_xd.size();
        size_t cols = matrix_xd[0].size();
        Eigen::MatrixXd mat(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat(i, j) = matrix_xd[i][j];
            }
        }
        return mat;
    }

    // Function to convert vectorXd to vector
    vector<double> convert_vactorXd_to_stdVector(const Eigen::VectorXd& vec) {
        return std::vector<double>(vec.data(), vec.data() + vec.size());
    }


};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class interpolate {
public:
    interpolate(SetGrid& grid) : grid_inst(grid), N(grid.size) {}

    //chebyshev interpolation
    double cheby(const vector<double>& f, double y) {
        double var = 0.0;
        for (size_t j = 1; j <= N; ++j) {
            var += cardinal_fct(j, y) * f[static_cast<size_t>(N - j)];
        }
        return var;

    }


private:
    SetGrid& grid_inst;
    size_t N;

    double cheby_polynomial_2_kind(size_t j, double y) {
        if (j == 0) return 1;
        if (j == 1) return 2 * y;

        double U_n_minus_2 = 1;
        double U_n_minus_1 = 2 * y;
        double U_n = 0;

        for (size_t i = 2; i <= j; ++i) {
            U_n = 2 * y * U_n_minus_1 - U_n_minus_2;
            U_n_minus_2 = U_n_minus_1;
            U_n_minus_1 = U_n;
        }

        return U_n;
    }

    //function to calculate the cardinal functions for the chebyshev expansion for [-1,1]. 
    double cardinal_fct_oneToMinusOne(size_t j, double y) {//j is for the order of the cardinal fct
        vector<double> z = grid_inst.z;

        if (y == z[j - 1]) return 1.0;

        double aux = (j == 1 || j == N) ? 2.0 : 1.0;
        return pow(-1, j) * (1 - y * y) * (N - 1) * cheby_polynomial_2_kind(N - 2, y) / (aux * pow(N - 1, 2) * (y - z[j - 1]));
    }


    //function to calculate the  cardinal functions for the chebyshev expansion for [0,1].
    double cardinal_fct(size_t j, double y) {//j is for the order of the cardinal fct
        return cardinal_fct_oneToMinusOne(j, 2.0 * y - 1.0);
    }

};


/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

class find_horizon {
public:

    double run(vector<vector<vector<double>>>& vals, const double& ksi, double& initial_guess, double& tolerance, int& max_iterations) {
        double x_n = initial_guess;

        for (size_t i = 0; i < static_cast<size_t>(max_iterations); ++i) {
            double f_xn = (1.0 / (2.0 * pow(x_n, 2))) + pow(x_n, 2) * Interp.cheby(vals[2][0], x_n) + (ksi / x_n) +
                0.5 * pow(ksi, 2);
            double df_xn = -(1.0 / pow(x_n, 3)) + 2.0 * x_n * Interp.cheby(vals[2][0], x_n) - (ksi / pow(x_n, 2)) +
                pow(x_n, 2) * Interp.cheby(vals[2][1], x_n);

            if (abs(f_xn) < tolerance) {return x_n;}
            x_n = x_n - (f_xn / df_xn);
        }
        std::cerr << "While looking for horizon position, max number of iterations is reached." << std::endl;//unnecessary
        return x_n;
    }


    find_horizon(SetGrid& grid) : Interp(grid) {}


private:

    interpolate Interp;

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class radial_shift {

public:
    
    //updates ksi and returns the change in ksi.
    double run(double& uh, double& desired_uh, double& ksi) {
        double change_in_Ksi = -(1.0 / (desired_uh)) + (1.0 / uh);
        ksi += change_in_Ksi;
        return change_in_Ksi;
    }
};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
class initialize {
public:

    initialize(SetGrid& grid, vector<vector<vector<double>>>& vals, double Beta, double Omega, double U0, double& ksi) : grid_inst(grid),
         solver(grid), horizon_finder(grid), beta(Beta), omega(Omega), u0(U0) {

        for (size_t i = 0; i < grid_inst.size; ++i) {
            vals[0][0][i] = aux_Bs0(ksi, grid_inst.u[i]);
        }

        vals[0][1] = multiplyMatrixVector(grid_inst.D, vals[0][0]);
        vals[0][2] = multiplyMatrixVector(grid_inst.D2, vals[0][0]);
    }

    void fix_horizon(vector<vector<vector<double>>>& vals, double& ksi, double& a4, double& uh, double& uh_desired,
        double& precision, double& tolerance, int& max_iterations) {
        int count = 0;
        while (count == 0 || abs(uh - uh_desired) > precision) {
            count += 1;
            // Run the solver
            solver.run(2, vals, ksi, a4);

            //Now find horizon
            uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
            
            if (abs(uh - uh_desired) > precision) {
                shift.run(uh, uh_desired, ksi);
                update(vals, ksi);
            }

        }

        cout << count - 1 << " iterative radial shilfts executed at first time slice. Now uh= " << uh << endl;
    }

 
private:
    double beta;
    double omega;
    double u0;
    SetGrid& grid_inst;
    solve solver;
    find_horizon horizon_finder;
    radial_shift shift;
   

    //define the function Bs at the first time slice for ksi=0
    double Bs0(double uu) {
        return beta * exp(-pow((uu - u0), 2) / pow(omega, 2));
    }

    double aux_Bs0(const double& ksi, double uu) {
        return pow(1.0 / (1 + uu * ksi), 4) * Bs0(uu / (1.0 + ksi * uu));
    }

    void update(vector<vector<vector<double>>>& vals, double& ksi) {
        //transform Bs to new coordinate system
        for (size_t i = 0; i < grid_inst.size; ++i) {
            vals[0][0][i] = aux_Bs0(ksi, grid_inst.u[i]);
        }
        vals[0][1] = multiplyMatrixVector(grid_inst.D, vals[0][0]);
        vals[0][2] = multiplyMatrixVector(grid_inst.D2, vals[0][0]);

    }

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class rk4_Bs_ksi_step {
public:

    rk4_Bs_ksi_step(SetGrid& grid):grid_inst(grid), interp(grid){}

    //run returns Bs and ksi at next time slice
     pair<vector<double>, double> run(vector<vector<vector<double>>>& vals, double& ksi, double& a4, solve& solver, double& uh, double dt) {

         vector<double> Bs_new(grid_inst.size);
         double ksi_new;

        //get the k stages for Bs and ksi
         pair<vector<vector<double>>, vector<double>> k = compute_k(vals, ksi, a4, solver, uh, dt);

        for (size_t j = 0; j < grid_inst.size; ++j) {
            Bs_new[j] = vals[0][0][j] +  (dt / 6.0) * ( k.first[0][j] + 2.0 * k.first[1][j] + 2.0 * k.first[2][j] + k.first[3][j]);
        }
        ksi_new = ksi + (dt / 6.0) * (k.second[0] + 2.0 * k.second[1] + 2.0 * k.second[2] + k.second[3]);

        return std::make_pair(Bs_new, ksi_new);
    }

private:
    SetGrid& grid_inst;
    interpolate interp;
   
    double dt_ksi( vector<vector<vector<double>>>& vals, double& ksi, double& uh){
        return (2.0 + 2.0 * pow(uh, 4) * interp.cheby(vals[4][0], uh) + pow(uh, 8) * pow(interp.cheby(vals[3][0], uh), 2) +
            4.0 * uh * ksi + 2.0 * pow(uh, 2) * pow(ksi, 2)) / (4.0 * pow(uh, 2)); // dt_ksi is time derivative of ksi.
    }

    

    vector<double> dt_Bs(vector<vector<vector<double>>>& vals, double& ksi, double& dt_Ksi){
        vector<double> D(grid_inst.size);

        D[0] = 0.5 * (8.0 * vals[0][0][0] * ksi + 2.0 * vals[3][1][0] + 5.0 * vals[0][1][0]);
        for (size_t i = 1; i < grid_inst.size; ++i) {
            D[i] = (1.0 / (2.0 * grid_inst.u[i])) * (2.0 * vals[3][0][i] + (1.0 + pow(grid_inst.u[i], 4) * vals[4][0][i]
                + 2.0 * grid_inst.u[i] * ksi + pow(grid_inst.u[i], 2) * pow(ksi, 2) -
                2.0 * pow(grid_inst.u[i], 2) * dt_Ksi) * (4.0 * vals[0][0][i] + grid_inst.u[i] * vals[0][1][i]));
        }
        return D;
    }


    pair<vector<vector<double>>, vector<double>> compute_k(vector<vector<vector<double>>>& vals, double& ksi, double& a4, solve& solver, double& uh, double&dt) {
        /*The pair contains a matrix for k_Bs and a vector for k_ksi */
        vector<double> k_ksi(4);
        vector<vector<double>> k_Bs(4, vector<double>(grid_inst.size ));
        
        //compute k1
        k_ksi[0] = dt_ksi(vals, ksi, uh);
        k_Bs[0] = dt_Bs(vals, ksi, k_ksi[0]);
        
        //compute k2
        double  ksi_temp =ksi + 0.5 * dt * k_ksi[0];
        vector<vector<vector<double>>> vals_temp = vals;
        for (size_t i=0; i<grid_inst.size; ++i){
            vals_temp[0][0][i] +=  dt * 0.5 * k_Bs[0][i];
        }
        vals_temp[0][1] = multiplyMatrixVector(grid_inst.D, vals_temp[0][0]);
        vals_temp[0][2] = multiplyMatrixVector(grid_inst.D2, vals_temp[0][0]);
        solver.run(4, vals_temp, ksi_temp, a4);

        k_ksi[1] = dt_ksi(vals_temp, ksi_temp, uh);
        k_Bs[1] = dt_Bs(vals_temp, ksi_temp, k_ksi[1]);

        //compute k3
        ksi_temp = ksi + 0.5 * dt * k_ksi[1];
        vals_temp = vals;
        for (size_t i=0; i < grid_inst.size; ++i) {
            vals_temp[0][0][i] += 0.5 * dt * k_Bs[1][i];
        }
        vals_temp[0][1] = multiplyMatrixVector(grid_inst.D, vals_temp[0][0]);
        vals_temp[0][2] = multiplyMatrixVector(grid_inst.D2, vals_temp[0][0]);
        solver.run(4, vals_temp, ksi_temp, a4);
        
        k_ksi[2] = dt_ksi(vals_temp, ksi_temp, uh); 
        k_Bs[2] = dt_Bs(vals_temp, ksi_temp, k_ksi[2]);

        //compute k4
        ksi_temp = ksi + dt * k_ksi[2];
        vals_temp = vals;
        for (size_t i=0; i < grid_inst.size; ++i) {
            vals_temp[0][0][i] += dt * k_Bs[2][i];
        }
        vals_temp[0][1] = multiplyMatrixVector(grid_inst.D, vals_temp[0][0]);
        vals_temp[0][2] = multiplyMatrixVector(grid_inst.D2, vals_temp[0][0]);
        solver.run(4, vals_temp, ksi_temp, a4);

        k_ksi[3] = dt_ksi(vals_temp, ksi_temp, uh);
        k_Bs[3] = dt_Bs(vals_temp, ksi_temp, k_ksi[3]);


        return std::make_pair(k_Bs, k_ksi);

    }

};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class adaptive_rk4_Bs_ksi {
public:
    adaptive_rk4_Bs_ksi(SetGrid& grid): RK4_Bs_ksi_step(grid) {}
    void run(SetGrid& grid, vector<vector<vector<double>>>& vals, double& ksi, double& a4, solve& solver, double& uh, double& t, double& t_f, double& dt,
        double& precision) {

        // Perform a full RK4 step for Bs and ksi
        pair<vector<double>, double> result_full_step/*containes a pair of Bs_new and ksi_new */ = RK4_Bs_ksi_step.run(vals, ksi, a4, solver, uh, dt);
       

        // Perform two 1/2-step RK4 steps for Bs and ksi
        pair<vector<double>, double> result_half_1 = RK4_Bs_ksi_step.run(vals, ksi,a4, solver,  uh, dt/2.0);

        vals_half_1 = vals;
        for (size_t i = 0; i < grid.size; ++i) {
            vals_half_1[0][0][i] = result_half_1.first[i];
        }
        vals_half_1[0][1] = multiplyMatrixVector(grid.D, vals_half_1[0][0]);
        vals_half_1[0][2] = multiplyMatrixVector(grid.D2, vals_half_1[0][0]);
        solver.run(4, vals_half_1, result_half_1.second, a4);


        pair<vector<double>, double> result_half_2 = RK4_Bs_ksi_step.run(vals_half_1, result_half_1.second, a4, solver, uh,  dt / 2.0);

        
        //calculate next time step  
        dt = dt * pow((pow(10.0, -precision) / compute_error(result_full_step, result_half_2)  ), 1.0 / 4.0);
        
        // Prevent the timestep from exceeding t_f
        if (t + dt >= t_f) {
            dt = t_f - t;
        }
        
        //update Bs and ksi with their values on the next time slice 
        vals[0][0] = result_full_step.first;
        ksi = result_full_step.second;

        t += dt;
    
    
    }


private:
    rk4_Bs_ksi_step RK4_Bs_ksi_step;
    vector<vector<vector<double>>> vals_half_1;
    

    double compute_error(pair<vector<double>, double>& result_1, pair<vector<double>, double>& result_2) {
        double err = 0.0;
        for (size_t i = 0; i < result_1.first.size(); ++i) {
            err = std::max(err, std::abs(result_1.first[i] - result_2.first[i]));
        }
        err = std::max(err, std::abs(result_1.second - result_2.second));

        return err;
    }


};



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Num_Routine {
private:
    
    int number_slices;
    double t;
    double deltaP_over_Peq;

    interpolate interp;
    adaptive_rk4_Bs_ksi adaptive_RK4_Bs_ksi;
    radial_shift shift;
    solve solver;
    find_horizon horizon_finder;
    double change_in_ksi;
    vector<double> aux_vector;

public:
    vector<double> horizonlist;
    vector<double> timelist;
    vector<double> deltaP_over_Peq_list;

    vector<vector<double>> Ss_list;//contains Ss at all times
    vector<vector<double>> Sdots_list;
    vector<vector<double>> Bdots_list;
    vector<vector<double>> As_list;
    vector<vector<double>> Bs_list;
    vector<vector<double>> B_over_u3_list;
    vector<double> ksi_list;
  

    Num_Routine(SetGrid& grid, double&t_0, double& t_f, double& dt, vector<vector<vector<double>>>& vals, double& ksi, double& a4,
        double& uh,  initialize& Initialize,  double uh_desired,
        double& precision/* for adaptive RK4*/ , double& tolerance/*for horizon finder*/, int& max_iterations/*for horizon finder*/):
        t(t_0), interp(grid), number_slices(0),
          adaptive_RK4_Bs_ksi(grid), solver(grid), horizon_finder(grid), aux_vector(grid.size), change_in_ksi(0.0){
        
        
        while (t <= t_f){
            number_slices += 1;
            
            if (number_slices % 100 == 0) {
                solver.run(2, vals, ksi, a4);
                uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
                

                change_in_ksi=shift.run(uh, uh_desired, ksi);
            }

            solver.run(4, vals, ksi, a4);
            
            uh = horizon_finder.run(vals, ksi, uh, tolerance, max_iterations);
        
           
            horizonlist.push_back(uh);
            ksi_list.push_back(ksi);
            timelist.push_back(t);

            Bs_list.push_back(vals[0][0]);
            
            vector<double> last_B_over_u3(grid.size);

            // component-wise multiplication of 2 vectors
            std::transform(grid.u.begin(), grid.u.end(), vals[0][0].begin(), last_B_over_u3.begin(),
                [](double a, double b) { return a * b; });

            B_over_u3_list.push_back(last_B_over_u3);
            Ss_list.push_back(vals[1][0]);
            Sdots_list.push_back(vals[2][0]);
            Bdots_list.push_back(vals[3][0]);
            As_list.push_back(vals[4][0]);
            

            deltaP_over_Peq = (12.0 / pow(M_PI, 2)) * vals[0][0][0] / a4;
            deltaP_over_Peq_list.push_back(deltaP_over_Peq);
            cout << "time slice: " << number_slices << " , t: " << t << " , dt: " << dt << " , deltaP_over_Peq: " << deltaP_over_Peq 
                <<" , uh: "<< uh<<endl;

            //calculates Bs and ksi at next time slice, finds the next timestep based on estimation of error and updates t
            if (t < t_f) {
                adaptive_RK4_Bs_ksi.run(grid, vals, ksi, a4, solver, uh, t, t_f, dt, precision);

                vals[0][1] = multiplyMatrixVector(grid.D, vals[0][0]);
                vals[0][2] = multiplyMatrixVector(grid.D2, vals[0][0]);
            }
            else { break; }
        }        
    }

};

/////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
class print {

public:
    print(vector<vector<vector<double>>>& vals, Num_Routine& num_routine, SetGrid& grid, int Precision)
        : precision(Precision) {

        writeDeltaPOverPeq(num_routine);
        writeMatrixToFile(num_routine.Bs_list, "Bs.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.Ss_list, "Ss.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.Sdots_list, "Sdots.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.Bdots_list, "Bdots.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.As_list, "As.dat", num_routine.timelist, grid.u);
        writeMatrixToFile(num_routine.B_over_u3_list, "B_over_u3.dat", num_routine.timelist, grid.u);

    }

private:
    int precision;

    void writeDeltaPOverPeq(Num_Routine& num_routine) {
        std::ofstream outFile("deltaP_over_Peq.dat");
        if (!outFile) {
            throw std::runtime_error("Unable to open deltaP_over_Peq.dat for writing!");
        }
        //set precision
        outFile << std::fixed << std::setprecision(precision);
        
        for (size_t i = 0; i < num_routine.timelist.size(); ++i) {
            outFile << num_routine.timelist[i] << " " << num_routine.deltaP_over_Peq_list[i] << endl;
        }
    }

    void writeMatrixToFile(vector<vector<double>>& matrix, string filename,
        vector<double>& t_values, vector<double>& u_values) {
        std::ofstream ofs(filename); 
        if (!ofs) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        ofs << std::fixed << std::setprecision(precision);
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                ofs << t_values[i] << " " << u_values[j] << " " << matrix[i][j] << "\n";
            }
            ofs << "\n"; 
        }
    }
};




int main() {

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << std::fixed << std::setprecision(10);//delete me

    double ksi=0.0;
    int number_grid_pts = 25;

    SetGrid grid(number_grid_pts); 
    vector<vector<vector<double>>> vals(5, vector<vector<double>>(3, vector<double>(grid.size)));

    double a4 = -1.0; //is twice a4 in Chesler and Yaffe convention.  

    // Bs at first time slice is defined (for ksi=0) as beta * exp(-pow((uu - u0), 2) / pow(omega, 2) ). 
    // Choose now values of beta, omega then u0.  
    initialize Initialize(grid, vals, 5.0, 0.15, 0.25, ksi);


    //fix horizon at some position uh_desired
    double uh=1.0;//initial guess for position of apparent horizon
    double uh_desired = 1.0;
    double precision_fix_horizon=1e-5;//uh is fixed to uh_desired up to precision_fix_horizon
    double tolerance = 1e-8;//for horizon finder
    int max_iterations = 10000;//for horizon finder

    Initialize.fix_horizon(vals, ksi, a4 , uh,  uh_desired, precision_fix_horizon, tolerance, max_iterations);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    double t_0 = 0.0; // Initial time
    double dt = 1 / (6.0 * pow(number_grid_pts, 2));  // Time step
    double t_f = 4.0; // Final time
    double precision_RK4 = 7;//adjust timestep to achieve a relative precision of 10^−precision_RK4

    Num_Routine num_routine(grid, t_0, t_f, dt, vals, ksi, a4,  uh, Initialize, uh_desired, precision_RK4, tolerance, max_iterations);

    auto end_numerics = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration_numerics = end_numerics - start;

    
    //Print out results in seperate dat files
    print Print(vals, num_routine, grid, 17);

    //print horiozn list
/*    std::ofstream outFile("horizon list.dat");
    if (!outFile) {
        throw std::runtime_error("Unable to open horizon list.dat for writing!");
    }
    //set precision
    outFile << std::fixed << std::setprecision(17);

    for (size_t i = 0; i < num_routine.timelist.size(); ++i) {
        outFile << num_routine.timelist[i] << " " << num_routine.horizonlist[i] << endl;
    }

    */

    // Calculate the program execution time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Numerical computations time, before printing out result to dat files: " << duration_numerics.count() << " seconds" << std::endl;
    std::cout << "Program execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}