#include <casadi/casadi.hpp>
#include "model_interface.hpp"
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

using namespace std;
casadi::DM EigenMatrixToCasadiMatrix(Eigen::MatrixXd eigen_matrix);
Eigen::MatrixXd CasadiMatrixToEigenMatrix(casadi::DM casadi_matrix);

int main()
{
  // ---------------------------------------------------------------------
  // Create a model based on a URDF file
  // ---------------------------------------------------------------------
//  std::string urdf_filename = "../urdf2model/models/atlas/urdf/atlas.urdf";
  std::string urdf_filename = "/home/kdyun/Desktop/urdf2modelcasadi/urdf2model/models/Hound_new_heavy/Hound_model.urdf";
  // Instantiate a Serial_Robot object called robot_model
  mecali::Serial_Robot robot_model;
  // Define (optinal) gravity vector to be used
  Eigen::Vector3d gravity_vector(0, 0, -9.81);
  // Create the model based on a URDF file
  robot_model.import_floating_base_model(urdf_filename, gravity_vector, true, true);

  // Print some information related to the imported model (boundaries, frames, DoF, etc)
  robot_model.print_model_data();

  // ---------------------------------------------------------------------
  // Set functions for robot dynamics and kinematics
  // ---------------------------------------------------------------------
  // Set function for forward dynamics
  //   casadi::Function fwd_dynamics = robot_model.forward_dynamics();
  // Set function for inverse dynamics
  // casadi::Function inv_dynamics = robot_model.inverse_dynamics();
  // Set function for forward kinematics
  // std::vector<std::string> required_Frames = {"Actuator1", "Actuator2", "Actuator3", "Actuator4", "Actuator5", "Actuator6", "Actuator7", "EndEffector" };

  // casadi::Function fkpos_ee = robot_model.forward_kinematics("position", "EndEffector");
  // casadi::Function fkrot_ee = robot_model.forward_kinematics("rotation", "EndEffector");
  // casadi::Function fk_ee    = robot_model.forward_kinematics("transformation", "EndEffector");
  casadi::Function fk = robot_model.forward_kinematics("position");
  casadi::Function fd = robot_model.forward_dynamics();
  casadi::Function id = robot_model.inverse_dynamics();

  casadi::Function J_fd = robot_model.forward_dynamics_derivatives("jacobian");
  casadi::Function J_id = robot_model.inverse_dynamics_derivatives("jacobian");

  casadi::Function mass_matrix = robot_model.mass_matrix();
  casadi::Function mass_inverse_matrix = robot_model.mass_inverse_matrix();
  casadi::Function coriolis_matrix = robot_model.coriolis_matrix();
  casadi::Function generalized_gravity = robot_model.generalized_gravity();

  casadi::Function joint_torque_regressor = robot_model.joint_torque_regressor();
  casadi::Function joint_torque_regressor_inertia_parameter = robot_model.joint_torque_regressor_inertia_parameter();
  casadi::Function inverse_dynamics_inertia_parameter = robot_model.inverse_dynamics_inertia_parameter();

  robot_model.generate_json("robot.json");

  // ---------------------------------------------------------------------
  // Generate (or save) a function
  // ---------------------------------------------------------------------
  // Code-generate or save a function
  // If you use options, you can set if you want to C-code-generate the function, or just save it as "second_function.casadi" (which can be loaded afterwards using casadi::Function::load("second_function.casadi"))
  mecali::Dictionary codegen_options;
  codegen_options["c"] = false;
  codegen_options["save"] = true;

  mecali::generate_code(fk, "fk", codegen_options);
  mecali::generate_code(fd, "fd", codegen_options);
  mecali::generate_code(id, "id", codegen_options);

  mecali::generate_code(J_fd, "J_fd", codegen_options);
  mecali::generate_code(J_id, "J_id", codegen_options);

  mecali::generate_code(mass_matrix, "mass_matrix", codegen_options);
  mecali::generate_code(mass_inverse_matrix, "mass_inverse_matrix", codegen_options);
  mecali::generate_code(coriolis_matrix, "coriolis_matrix", codegen_options);
  mecali::generate_code(generalized_gravity, "generalized_gravity", codegen_options);

  mecali::generate_code(joint_torque_regressor, "joint_torque_regressor", codegen_options);
  mecali::generate_code(joint_torque_regressor_inertia_parameter, "joint_torque_regressor_inertia_parameter", codegen_options);
  mecali::generate_code(inverse_dynamics_inertia_parameter, "inverse_dynamics_inertia_parameter", codegen_options);


  // ---------------------------------------------------------------------
  // load functions
  // ---------------------------------------------------------------------

  casadi::Function fk_test = casadi::Function::load("fk.casadi");
  casadi::Function fd_test = casadi::Function::load("fd.casadi");
  casadi::Function id_test = casadi::Function::load("id.casadi");

  casadi::Function J_fd_test = casadi::Function::load("J_fd.casadi");
  casadi::Function J_id_test = casadi::Function::load("J_id.casadi");

  casadi::Function mass_matrix_test = casadi::Function::load("mass_matrix.casadi");
  casadi::Function mass_inverse_matrix_test = casadi::Function::load("mass_inverse_matrix.casadi");
  casadi::Function coriolis_matrix_test = casadi::Function::load("coriolis_matrix.casadi");
  casadi::Function generalized_gravity_test = casadi::Function::load("generalized_gravity.casadi");

  casadi::Function joint_torque_regressor_test = casadi::Function::load("joint_torque_regressor.casadi");
  casadi::Function joint_torque_regressor_inertia_parameter_test = casadi::Function::load("joint_torque_regressor_inertia_parameter.casadi");
  casadi::Function inverse_dynamics_inertia_parameter_test = casadi::Function::load("inverse_dynamics_inertia_parameter.casadi");

  std::cout << "symbolic function" << std::endl;
  std::cout << fk_test << std::endl;
  std::cout << fd_test << std::endl;
  std::cout << id_test << std::endl;
  std::cout << J_fd_test << std::endl;
  std::cout << J_id_test << std::endl;
  std::cout << mass_matrix_test << std::endl;
  std::cout << mass_inverse_matrix_test << std::endl;
  std::cout << coriolis_matrix_test << std::endl;
  std::cout << generalized_gravity_test << std::endl;
  std::cout << joint_torque_regressor_test << std::endl;
  std::cout << joint_torque_regressor_inertia_parameter_test << std::endl;
  std::cout << inverse_dynamics_inertia_parameter_test << std::endl;
  std::cout << std::endl;

  std::cout << "Test result: "<< std::endl;
  std::vector<double> tau_vec = {0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,};
//  std::vector<double> q_vec = {0, 0, 0,  1, 0, 0, 0,  0.0, 0.0, 0.0,  0, 0, 0,  0, 0, 0,  0, 0, 1,};
  std::vector<double> q_vec = {0, 0, 0,  1, 0, 0, 0,  0.1, 0.1, 0.1,  0, 0, 0,  0, 0, 0,  0, 0, 1,};
//  std::vector<double> dq_vec = {0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1,};
  std::vector<double> dq_vec = {0.1, 0.2, 0.3,  0.4, 0.5, 0.6,  0.7, 0.8, 0.9,  1.0, 1.10, 1.20,  1.30, 1.40, 1.50,  1.60, 1.70, 1.8,};
  std::vector<double> ddq_vec = {0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 0, 1,};
  std::vector<double> inertia_param(130,0);

//  casadi::DM ddq = fd_test(casadi::DMVector {q_vec,dq_vec,tau_vec})[0];
//  std::cout << "ddq" << ddq << std::endl;
  casadi::DM tau = id_test(casadi::DMVector {q_vec,dq_vec,ddq_vec})[0];

//  casadi::DM regressor = joint_torque_regressor_test(casadi::DMVector {q_vec,dq_vec,ddq})[0];
  casadi::DM regressor = joint_torque_regressor_test(casadi::DMVector {q_vec,dq_vec,ddq_vec})[0];
  casadi::DM regressor_inertia_param = joint_torque_regressor_inertia_parameter_test(casadi::DMVector {q_vec,dq_vec,ddq_vec,inertia_param})[0];
  casadi::DM tau_param = inverse_dynamics_inertia_parameter_test(casadi::DMVector {q_vec,dq_vec,ddq_vec,inertia_param})[0];

  std::cout << regressor.size() << std::endl;
//  std::cout << regressor << std::endl;
//  std::cout << regressor_inertia_param.size() << std::endl;
//  std::cout << regressor_inertia_param << std::endl;


//  std::cout << CasadiMatrixToEigenMatrix(regressor) << std::endl;

//  std::cout << "Row 0:\n" << regressor(casadi::Slice(0, 1), casadi::Slice()).size() << std::endl;
//  std::cout << regressor(casadi::Slice(0, 1), casadi::Slice()) << std::endl;
//  std::cout << "Col 0:\n" << regressor(casadi::Slice(), casadi::Slice(0,1)).size() << std::endl;
//  for (auto col=0; col != 130; col++){
//    std::cout << "Col"<< col <<":\n" << regressor(casadi::Slice(), casadi::Slice(col,col+1)) << std::endl;
//  }
//  std::cout << "barycentric parameters: " << robot_model.barycentric_params.transpose() << std::endl;
  std::cout << "barycentric parameters: " << robot_model.barycentric_params.head((robot_model.n_joints-1)*10).transpose() << std::endl;
//  auto torque = mtimes(regressor,EigenMatrixToCasadiMatrix(robot_model.barycentric_params.head((robot_model.n_joints-1)*10))) ;
  auto torque = (CasadiMatrixToEigenMatrix(regressor)*robot_model.barycentric_params.head((robot_model.n_joints-1)*10)).transpose() ; /// why wrong?

  std::cout << "tau   : " << tau << std::endl;
  std::cout << "result: " << torque<< std::endl;

//  auto test = robot_model._casadi_model.inertias[1].mass();
//  std::cout << "test   : " << test.type_name() << std::endl;

  raisim::ArticulatedSystem *Hound = new raisim::ArticulatedSystem(urdf_filename);

//  std::cout << "gc " <<  casadi::DMVector {q_vec}[0]<< std::endl;
//  std::cout << "gv " <<  casadi::DMVector {dq_vec}[0] << std::endl;
//  std::cout << "ga " <<  casadi::DMVector {ddq_vec}[0] << std::endl;
  std::cout << "gc " <<  CasadiMatrixToEigenMatrix(casadi::DMVector {q_vec}[0]).transpose()<< std::endl;
  std::cout << "gv " <<  CasadiMatrixToEigenMatrix(casadi::DMVector {dq_vec}[0]).transpose() << std::endl;
  std::cout << "ga " <<  CasadiMatrixToEigenMatrix(casadi::DMVector {ddq_vec}[0]).transpose() << std::endl;
  Hound->setState(CasadiMatrixToEigenMatrix(casadi::DMVector {q_vec}[0]),CasadiMatrixToEigenMatrix(casadi::DMVector {dq_vec}[0]));
  Eigen::MatrixXd M = Hound->getMassMatrix().e();
  Eigen::VectorXd h = Hound->getNonlinearities(raisim::Vec<3>{0,0,-9.81}).e();
//  Eigen::VectorXd h2 = Go1_Dynamic->getNonlinearities(raisim::Vec<3>{0,0,0}).e();
//  Eigen::VectorXd h3 = Go1_Dynamic->getNonlinearities(raisim::Vec<3>{0,0,9.81}).e();
  std::cout << "M " <<  M << std::endl;
  std::cout << "h " <<  h.transpose() << std::endl;
  std::cout << "M*ga+h " <<  (M*CasadiMatrixToEigenMatrix(casadi::DMVector {ddq_vec}[0])+h).transpose() << std::endl;
  std::cout << tau << std::endl;



}
casadi::DM EigenMatrixToCasadiMatrix(Eigen::MatrixXd eigen_matrix){
  casadi::DM casadi_matrix = casadi::DM::zeros(eigen_matrix.rows(), eigen_matrix.cols());
  for (int i = 0; i < eigen_matrix.rows(); ++i) {
    for (int j = 0; j < eigen_matrix.cols(); ++j) {
      casadi_matrix(i, j) = eigen_matrix(i, j);
    }
  }
  return casadi_matrix;
}
Eigen::MatrixXd CasadiMatrixToEigenMatrix(casadi::DM casadi_matrix){
  Eigen::MatrixXd eigen_matrix(casadi_matrix.rows(), casadi_matrix.columns());
  for (int i = 0; i < casadi_matrix.rows(); ++i) {
    for (int j = 0; j < casadi_matrix.columns(); ++j) {
      eigen_matrix(i, j) = static_cast<double>(casadi_matrix(i, j));
    }
  }
  return eigen_matrix;
}