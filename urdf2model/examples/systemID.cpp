#include "model_interface.hpp"
#include <Eigen/Dense>
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"


void loadData(Eigen::MatrixXd& gc_matrix, Eigen::MatrixXd& gv_matrix, Eigen::MatrixXd& ga_matrix, Eigen::MatrixXd& tau_matrix);
Eigen::VectorXd RaisimGcToPinocchioGc(Eigen::VectorXd raisim_gc);
Eigen::VectorXd RaisimGvToPinocchioGv(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_gv);
Eigen::VectorXd RaisimGaToPinocchioGa(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_gv,Eigen::VectorXd raisim_ga);
Eigen::VectorXd RaisimTauToPinocchioTau(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_tau);
Eigen::MatrixXd RaisimMatToPinocchioMat(Eigen::VectorXd raisim_gc,Eigen::MatrixXd raisim_mat);
Eigen::VectorXd PinocchioGvToRaisimGv(Eigen::VectorXd pin_gc,Eigen::VectorXd pin_gv);

Eigen::Matrix3d skewMat(Eigen::Vector3d vec);
Eigen::VectorXd getBarycentricParams(pinocchio::Model model);
Eigen::VectorXd vech(Eigen::Matrix3d inertia);
Eigen::Matrix3d invvech(Eigen::VectorXd vec);
Eigen::Matrix4d getPseudoInertia(Eigen::VectorXd vec);
bool checkPhysicalConsistency(Eigen::Matrix4d pseudoInertia);

template <typename T>
void print_indent(std::string var_name, T var_value,               int indent);
void print_indent(std::string var_name, Eigen::VectorXd var_value, int indent);

int main(){
  std::cout.precision(15);
  std::string urdf_filename = "/home/kdyun/Desktop/urdf2modelcasadi/urdf2model/models/go1/go1.urdf";

  Eigen::Vector3d gravity_vector(0, 0, -9.81);
//  Eigen::Vector3d gravity_vector(0, 0, 0);

  /// raisim model
  // raisim convention
  // q = [global_base_position, global_base_quaternion(w,x,y,z), joint_positions(FR,FL,RR,RL)]
  // v = [global_base_velocity_linear, global_base_velocity_angular, joint_velocities]
  // a = [global_base_acceleration_linear, global_base_acceleration_angular, joint_accelerations]

  raisim::ArticulatedSystem *Go1_raisim = new raisim::ArticulatedSystem(urdf_filename);

  /// pinocchio model
  // pinocchio convention  *local = joint frame
  // q = [global_base_position, global_base_quaternion(x,y,z,w), joint_positions(FL,FR,RL,RR)]
  // v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities]
  // a = [local_base_acceleration_linear, local_base_acceleration_angular, joint_accelerations]
  pinocchio::Model Go1_pin_model;
  pinocchio::urdf::buildModel(urdf_filename,pinocchio::JointModelFreeFlyer(),Go1_pin_model);

  Go1_pin_model.gravity.linear(gravity_vector);

  pinocchio::Data Go1_pin_data = pinocchio::Data(Go1_pin_model);
  Eigen::VectorXd barycentric_params = getBarycentricParams(Go1_pin_model);


  /// load Data
  Eigen::MatrixXd gc_matrix,gv_matrix,ga_matrix,tau_matrix;
  loadData(gc_matrix,gv_matrix,ga_matrix,tau_matrix);


  int dataLength = 10000;
  Eigen::MatrixXd W; Eigen::VectorXd w;
  W.setZero(dataLength*18,130); w.setZero(dataLength*18);


  Eigen::VectorXd parameter_est = barycentric_params*1.05;
  parameter_est.setZero();
  Eigen::VectorXd gradient; gradient.setZero(130);
  Eigen::VectorXd delta; delta.setZero(130);


  int num_epoch = 1;
  for (int j = 0; j < num_epoch; ++j) {
    for (int i = 0; i < 10000; ++i) {
//    std::cout << i << " -----------------------------------------" << std::endl;
      Eigen::VectorXd raisim_gc, raisim_gv, raisim_ga, raisim_tau;
      raisim_gc.setZero(19);
      raisim_gv.setZero(18);
      raisim_ga.setZero(18);
      raisim_tau.setZero(18);
      raisim_gc = gc_matrix.row(i);
      raisim_gv = gv_matrix.row(i);
      raisim_ga = ga_matrix.row(i);
      raisim_tau = tau_matrix.row(i);

      /// convert raisim to pinocchio
      Eigen::VectorXd pin_gc, pin_gv, pin_ga, pin_tau;
      pin_gc.setZero(19);
      pin_gv.setZero(18);
      pin_ga.setZero(18);
      pin_tau.setZero(18);
      pin_gc = RaisimGcToPinocchioGc(raisim_gc);
      pin_gv = RaisimGvToPinocchioGv(raisim_gc, raisim_gv);
      pin_ga = RaisimGaToPinocchioGa(raisim_gc, raisim_gv, raisim_ga);
      pin_tau = RaisimTauToPinocchioTau(raisim_gc, raisim_tau);
      pinocchio::normalize(Go1_pin_model, pin_gc);

//    // test EoM
//    pinocchio::crba(Go1_pin_model, Go1_pin_data, pin_gc, pinocchio::Convention::LOCAL); // fill only upper triangular
//    Go1_pin_data.M.triangularView<Eigen::StrictlyLower>() = Go1_pin_data.M.transpose().triangularView<Eigen::StrictlyLower>();
//    pinocchio::nonLinearEffects(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv); // run this after rnea?
////  std::cout << "pin error " <<  (Go1_pin_data.M*pin_ga+Go1_pin_data.nle-pin_tau).transpose() << std::endl;
//    if((Go1_pin_data.M * pin_ga + Go1_pin_data.nle - pin_tau).norm()>1e-8){
//      std::cout << "  pin error norm " << (Go1_pin_data.M * pin_ga + Go1_pin_data.nle - pin_tau).norm() << std::endl;
//    }

      // test regressor
      pinocchio::rnea( Go1_pin_model, Go1_pin_data, pin_gc, pin_gv, pin_ga );    // Call the Recursive Newton-Euler algorithm first
      pinocchio::computeJointTorqueRegressor(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv, pin_ga);
//    if((pin_tau-Go1_pin_data.jointTorqueRegressor*barycentric_params).norm()>1e-8) {
//      std::cout << "  joint torque regressor norm "<< (pin_tau - Go1_pin_data.jointTorqueRegressor * barycentric_params).norm() << std::endl;
//    }

      W.block(i*18,0,18,130)=Go1_pin_data.jointTorqueRegressor;
      w.segment(i*18,18)=pin_tau;


      /// Iterative Optimization
      // min_dx ||W*(x+dx)-w||^2 + gamma*|| dx ||^2
      // gradient 2*W^T*(W*(x+dx)-w) +2*gamma*dx = 0
      // dx = -(W^T*W+gamma*I)^(-1)*(W^T*W*x-W^T*w)
      // x = x+dx
      Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(Go1_pin_data.jointTorqueRegressor);
      auto rank = lu_decomp.rank();
      double gamma = 1;
      delta.setZero();
      /// 벡터 차원 선언 주의!!
      //    gradient = (Go1_pin_data.jointTorqueRegressor.transpose() * Go1_pin_data.jointTorqueRegressor).ldlt().solve(Go1_pin_data.jointTorqueRegressor.transpose() * pin_tau)-parameter_est; // Newton direction
//    gradient = Go1_pin_data.jointTorqueRegressor.transpose()*(Go1_pin_data.jointTorqueRegressor*(parameter_est+delta) - pin_tau) + delta;
      delta = -(Go1_pin_data.jointTorqueRegressor.transpose() * Go1_pin_data.jointTorqueRegressor+gamma*Eigen::MatrixXd::Identity(130,130)).ldlt().solve(Go1_pin_data.jointTorqueRegressor.transpose() * Go1_pin_data.jointTorqueRegressor*parameter_est - Go1_pin_data.jointTorqueRegressor.transpose() * pin_tau);
      if(i%100==0){
        std::cout << i <<" | param error:"<< (parameter_est-barycentric_params).norm()<< " | regression error:"<< (Go1_pin_data.jointTorqueRegressor*parameter_est-pin_tau).norm()<<std::endl;
        std::cout << "delta " <<delta.transpose()<<std::endl;
      }
      parameter_est+=delta;

    }
  }


/// Total optimization
  std::cout << "--------------------Total optimization--------------------------------" <<std::endl;

// min ||W*alpha-w||^2
// gradient 2*W^T(W*alpha-w)

std::cout << "("<< W.rows() << ","<< W.cols() <<")"<<std::endl;
std::cout << w.size() <<std::endl;
Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(W);
auto rank = lu_decomp.rank();
std::cout << "W rank " <<rank<<std::endl;



Eigen::VectorXd parameter_est1 = W.colPivHouseholderQr().solve(w); // QR decomposition
Eigen::VectorXd parameter_est2 = (W.transpose() * W).ldlt().solve(W.transpose() * w); // normal equation
//double gamma = 1e-9;
//Eigen::VectorXd parameter_est3 = (W.transpose() * W+gamma*Eigen::MatrixXd::Identity(130,130)).ldlt().solve(W.transpose() * w+gamma*(parameter_true)); // normal equation + regularization


std::cout << "--------------------validation--------------------------------" <<std::endl;
std::cout << "parameter true" << " | regression error:"<< (W*barycentric_params-w).norm()<<std::endl;
std::cout << (barycentric_params).transpose() <<std::endl;parameter_est2;

std::cout << "parameter_est " << " | param error:"<< (parameter_est-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est-w).norm()<<std::endl;
std::cout << (parameter_est).transpose()  <<std::endl;
//std::cout << (W*parameter_est1-w).transpose() <<std::endl;


std::cout << "parameter_est1 " << " | param error:"<< (parameter_est1-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est1-w).norm()<<std::endl;
std::cout << (parameter_est1).transpose()  <<std::endl;
//std::cout << (W*parameter_est1-w).transpose() <<std::endl;

std::cout << "parameter_est2 " << " | param error:"<< (parameter_est2-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est2-w).norm()<<std::endl;
std::cout << (parameter_est2).transpose()  <<std::endl;
//std::cout << (W*parameter_est2-w).transpose() <<std::endl;

//std::cout << "parameter_est3 " << " | param error:"<< (parameter_est3-parameter_true).norm()<< " | regression error:"<< (W*parameter_est3-w).norm()<< " | regularization error:"<< gamma*(parameter_est3-parameter_true*1.1).norm()<<std::endl;
//std::cout << (parameter_est3).transpose()  <<std::endl;
//std::cout << (W*parameter_est3-w).transpose() <<std::endl;


// check constraint violation !!!
  for (int j = 0; j < 13; ++j) {
    std::cout << "-------------------------------------" <<std::endl;
    std::cout << "Link "<<j <<" = "<< Go1_pin_model.names[j+1] <<std::endl;
    Eigen::VectorXd DynamicParameter = parameter_est1.segment(j*10,10);
//    Eigen::VectorXd DynamicParameter = parameter_est2.segment(j*10,10);
//    Eigen::VectorXd DynamicParameter = parameter_est3.segment(j*10,10);
    Eigen::VectorXd DynamicParameter_true = barycentric_params.segment(j*10,10);

    std::cout<< std::right << std::setw(20) <<"Inertial Properties "<<"||";
    std::cout<< std::right << std::setw(20) <<"Mass"<<"||"; // mass >0
    std::cout<< std::right << std::setw(12) <<"m cx";
    std::cout<< std::right << std::setw(12) <<"m cy";
    std::cout<< std::right << std::setw(12) <<"m cz"<<"||";
    std::cout<< std::right << std::setw(12) <<"Ixx";
    std::cout<< std::right << std::setw(12) <<"Ixy";
    std::cout<< std::right << std::setw(12) <<"Iyy";
    std::cout<< std::right << std::setw(12) <<"Ixz";
    std::cout<< std::right << std::setw(12) <<"Iyz";
    std::cout<< std::right << std::setw(12) <<"Izz"<<"||";
    std::cout<<std::endl;


    std::cout<<std::right << std::setw(20)<<"True Values "<<"||";
    std::cout<<std::right << std::setw(20)<<DynamicParameter_true(0)<<"||";
    std::cout<<std::right << std::setw(3)<<DynamicParameter_true.segment(1,3).transpose()<<"||";
    std::cout<<std::right << std::setw(3)<<DynamicParameter_true.segment(4,6).transpose()<<"||";
    std::cout<<std::endl;

    std::cout<<std::right << std::setw(20)<<"Estimated Values "<<"||";
    std::cout<<std::right << std::setw(20)<<DynamicParameter(0)<<"||";
    std::cout<<std::right << std::setw(3)<< DynamicParameter.segment(1,3).transpose()<<"||";
    std::cout<<std::right << std::setw(3)<< DynamicParameter.segment(4,6).transpose()<<"||";
    std::cout<<std::endl;

    std::cout<< checkPhysicalConsistency(getPseudoInertia(DynamicParameter_true)) <<std::endl;
    std::cout<< checkPhysicalConsistency(getPseudoInertia(DynamicParameter)) <<std::endl;

    std::cout << "-------------------------------------" <<std::endl;
  }
}



/*
int main(){
  std::cout.precision(15);
  std::string urdf_filename = "/home/kdyun/Desktop/urdf2modelcasadi/urdf2model/models/go1/go1.urdf";

  Eigen::Vector3d gravity_vector(0, 0, -9.81);
//  Eigen::Vector3d gravity_vector(0, 0, 0);

  /// raisim model
  // raisim convention
  // q = [global_base_position, global_base_quaternion(w,x,y,z), joint_positions(FR,FL,RR,RL)]
  // v = [global_base_velocity_linear, global_base_velocity_angular, joint_velocities]
  // a = [global_base_acceleration_linear, global_base_acceleration_angular, joint_accelerations]

  raisim::ArticulatedSystem *Go1_raisim = new raisim::ArticulatedSystem(urdf_filename);

  /// pinocchio model
  // pinocchio convention  *local = joint frame
  // q = [global_base_position, global_base_quaternion(x,y,z,w), joint_positions(FL,FR,RL,RR)]
  // v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities]
  // a = [local_base_acceleration_linear, local_base_acceleration_angular, joint_accelerations]
  pinocchio::Model Go1_pin_model;
  pinocchio::urdf::buildModel(urdf_filename,pinocchio::JointModelFreeFlyer(),Go1_pin_model);
  Go1_pin_model.gravity.linear(gravity_vector);
  pinocchio::Data Go1_pin_data = pinocchio::Data(Go1_pin_model);

//  std::vector<std::string> list_of_joints_to_lock_by_name = {}; // basically, base is locked as well
  std::vector<std::string> list_of_joints_to_lock_by_name = {"FL_roll_joint"}; // basically, base is locked as well
  std::vector<mecali::Index> list_of_joints_to_lock_by_id;
  for (std::vector<std::string>::const_iterator it = list_of_joints_to_lock_by_name.begin();it != list_of_joints_to_lock_by_name.end(); ++it){
    const std::string &joint_name = *it;
    if (Go1_pin_model.existJointName(joint_name)) // do not consider joint that are not in the model
      list_of_joints_to_lock_by_id.push_back(Go1_pin_model.getJointId(joint_name));
    else
      std::cout << "joint: " << joint_name << " does not belong to the model" << std::endl;
  }
  Eigen::VectorXd robot_configuration = pinocchio::neutral(Go1_pin_model);
  auto Go1_pin_model_reduced = pinocchio::buildReducedModel(Go1_pin_model,list_of_joints_to_lock_by_id,robot_configuration);
  pinocchio::Data Go1_pin_data_reduced = pinocchio::Data(Go1_pin_model_reduced);

  Eigen::VectorXd barycentric_params = getBarycentricParams(Go1_pin_model_reduced).head(10*Go1_pin_model_reduced.nv);


  std::cout << "\n----- Robot model information: " << std::endl;
  print_indent("Model name = ", Go1_pin_model_reduced.name, 38);
  print_indent("Size of configuration vector = ", Go1_pin_model_reduced.nq, 38);
  print_indent("Number of joints (with universe) = ", Go1_pin_model_reduced.njoints, 38);
  print_indent("Number of DoF = ", Go1_pin_model_reduced.nv, 38);
  print_indent("Number of bodies = ", Go1_pin_model_reduced.nbodies, 38);
  print_indent("Number of operational frames = ", Go1_pin_model_reduced.nframes, 38);
  print_indent("Gravity = ", Go1_pin_model_reduced.gravity, 38);
  print_indent("Joint torque bounds = ", Go1_pin_model_reduced.effortLimit, 38);
  print_indent("Joint configuration upper bounds = ", Go1_pin_model_reduced.upperPositionLimit, 38);
  print_indent("Joint configuration lower bounds = ", Go1_pin_model_reduced.lowerPositionLimit, 38);
  print_indent("Joint velocity bounds = ", Go1_pin_model_reduced.velocityLimit, 38);
  print_indent("Barycentric parameters = ", barycentric_params, 45);
  std::cout << std::endl;

  // std::cout << "\n----- Placement of each joint in the model: " << std::endl;
  std::cout << "-----Name of each joint in the model: " << std::endl;
  for (int k = 0; k < Go1_pin_model_reduced.njoints; ++k)
  {
    std::cout << std::setprecision(3) << std::left << std::setw(5) << k << std::setw(20) << Go1_pin_model_reduced.names[k] << std::setw(10) << std::endl; // << data.oMi[k].translation().transpose()
  }

  // std::cout << "\n----- Name of each frame in the model: " << std::endl;
  // for (int k=0 ; k<Go1_pin_model_reduced.n_frames ; ++k)
  // {
  //     std::cout << std::setprecision(3) << std::left << std::setw(5) <<  k  << std::setw(20) << Go1_pin_model_reduced._model.frames[k].name << std::endl;
  // }





  /// load Data
  Eigen::MatrixXd gc_matrix,gv_matrix,ga_matrix,tau_matrix;
  loadData(gc_matrix,gv_matrix,ga_matrix,tau_matrix);


  int dataLength = 10000;
  Eigen::MatrixXd W; Eigen::VectorXd w;
  W.setZero(dataLength*18,130); w.setZero(dataLength*18);


  Eigen::VectorXd parameter_est = barycentric_params*1.05;
  Eigen::VectorXd gradient; gradient.setZero(130);
  Eigen::VectorXd delta; delta.setZero(130);


  for (int i = 0; i < 10000; ++i) {
//    std::cout << i << " -----------------------------------------" << std::endl;
    Eigen::VectorXd raisim_gc, raisim_gv, raisim_ga, raisim_tau;
    raisim_gc.setZero(19);
    raisim_gv.setZero(18);
    raisim_ga.setZero(18);
    raisim_tau.setZero(18);
    raisim_gc = gc_matrix.row(i);
    raisim_gv = gv_matrix.row(i);
    raisim_ga = ga_matrix.row(i);
    raisim_tau = tau_matrix.row(i);

    /// convert raisim to pinocchio
    Eigen::VectorXd pin_gc, pin_gv, pin_ga, pin_tau;
    pin_gc.setZero(19);
    pin_gv.setZero(18);
    pin_ga.setZero(18);
    pin_tau.setZero(18);
    pin_gc = RaisimGcToPinocchioGc(raisim_gc);
    pin_gv = RaisimGvToPinocchioGv(raisim_gc, raisim_gv);
    pin_ga = RaisimGaToPinocchioGa(raisim_gc, raisim_gv, raisim_ga);
    pin_tau = RaisimTauToPinocchioTau(raisim_gc, raisim_tau);
    pinocchio::normalize(Go1_pin_model, pin_gc);

//    // test EoM
//    pinocchio::crba(Go1_pin_model, Go1_pin_data, pin_gc, pinocchio::Convention::LOCAL); // fill only upper triangular
//    Go1_pin_data.M.triangularView<Eigen::StrictlyLower>() = Go1_pin_data.M.transpose().triangularView<Eigen::StrictlyLower>();
//    pinocchio::nonLinearEffects(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv); // run this after rnea?
////  std::cout << "pin error " <<  (Go1_pin_data.M*pin_ga+Go1_pin_data.nle-pin_tau).transpose() << std::endl;
//    if((Go1_pin_data.M * pin_ga + Go1_pin_data.nle - pin_tau).norm()>1e-8){
//      std::cout << "  pin error norm " << (Go1_pin_data.M * pin_ga + Go1_pin_data.nle - pin_tau).norm() << std::endl;
//    }

    // test regressor
    pinocchio::rnea( Go1_pin_model, Go1_pin_data, pin_gc, pin_gv, pin_ga );    // Call the Recursive Newton-Euler algorithm first
    pinocchio::computeJointTorqueRegressor(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv, pin_ga);
//    if((pin_tau-Go1_pin_data.jointTorqueRegressor*barycentric_params).norm()>1e-8) {
//      std::cout << "  joint torque regressor norm "<< (pin_tau - Go1_pin_data.jointTorqueRegressor * barycentric_params).norm() << std::endl;
//    }

    W.block(i*18,0,18,130)=Go1_pin_data.jointTorqueRegressor;
    w.segment(i*18,18)=pin_tau;


    /// Iterative Optimization
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(Go1_pin_data.jointTorqueRegressor);
    auto rank = lu_decomp.rank();
    for (int j = 0; j < 10; ++j) {
      delta.setZero();
      //    gradient = (Go1_pin_data.jointTorqueRegressor.transpose() * Go1_pin_data.jointTorqueRegressor).ldlt().solve(Go1_pin_data.jointTorqueRegressor.transpose() * pin_tau)-parameter_est; // Newton direction
      gradient = Go1_pin_data.jointTorqueRegressor.transpose()*(Go1_pin_data.jointTorqueRegressor*parameter_est - pin_tau);
      delta = -1e-10*(Go1_pin_data.jointTorqueRegressor*parameter_est - pin_tau).norm()*gradient;
      parameter_est+=delta;
    }
    if(i%100==0){
      std::cout << i <<" | param error:"<< (parameter_est-barycentric_params).norm()<< " | regression error:"<< (Go1_pin_data.jointTorqueRegressor*parameter_est-pin_tau).norm()<<std::endl;
      std::cout << "W_iter rank " <<rank<<std::endl;

    }

  }

/// Total optimization
  std::cout << "--------------------Total optimization--------------------------------" <<std::endl;

// min ||W*alpha-w||^2
// gradient 2*W^T(W*alpha-w)

  std::cout << "("<< W.rows() << ","<< W.cols() <<")"<<std::endl;
  std::cout << w.size() <<std::endl;
  Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(W);
  auto rank = lu_decomp.rank();
  std::cout << "W rank " <<rank<<std::endl;



  Eigen::VectorXd parameter_est1 = W.colPivHouseholderQr().solve(w); // QR decomposition
  Eigen::VectorXd parameter_est2 = (W.transpose() * W).ldlt().solve(W.transpose() * w); // normal equation
//double gamma = 1e-9;
//Eigen::VectorXd parameter_est3 = (W.transpose() * W+gamma*Eigen::MatrixXd::Identity(130,130)).ldlt().solve(W.transpose() * w+gamma*(parameter_true)); // normal equation + regularization


  std::cout << "--------------------validation--------------------------------" <<std::endl;
  std::cout << "parameter true" << " | regression error:"<< (W*barycentric_params-w).norm()<<std::endl;
  std::cout << (barycentric_params).transpose() <<std::endl;

  std::cout << "parameter_est " << " | param error:"<< (parameter_est-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est-w).norm()<<std::endl;
  std::cout << (parameter_est).transpose()  <<std::endl;
//std::cout << (W*parameter_est1-w).transpose() <<std::endl;


  std::cout << "parameter_est1 " << " | param error:"<< (parameter_est1-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est1-w).norm()<<std::endl;
  std::cout << (parameter_est1).transpose()  <<std::endl;
//std::cout << (W*parameter_est1-w).transpose() <<std::endl;

  std::cout << "parameter_est2 " << " | param error:"<< (parameter_est2-barycentric_params).norm()<< " | regression error:"<< (W*parameter_est2-w).norm()<<std::endl;
  std::cout << (parameter_est2).transpose()  <<std::endl;
//std::cout << (W*parameter_est2-w).transpose() <<std::endl;

//std::cout << "parameter_est3 " << " | param error:"<< (parameter_est3-parameter_true).norm()<< " | regression error:"<< (W*parameter_est3-w).norm()<< " | regularization error:"<< gamma*(parameter_est3-parameter_true*1.1).norm()<<std::endl;
//std::cout << (parameter_est3).transpose()  <<std::endl;
//std::cout << (W*parameter_est3-w).transpose() <<std::endl;


// check constraint violation !!!
  for (int j = 0; j < 13; ++j) {
    std::cout << "-------------------------------------" <<std::endl;
    std::cout << "Link "<<j <<" = "<< Go1_pin_model.names[j+1] <<std::endl;
    Eigen::VectorXd DynamicParameter = parameter_est1.segment(j*10,10);
//    Eigen::VectorXd DynamicParameter = parameter_est2.segment(j*10,10);
//    Eigen::VectorXd DynamicParameter = parameter_est3.segment(j*10,10);
    Eigen::VectorXd DynamicParameter_true = barycentric_params.segment(j*10,10);

    std::cout<< std::right << std::setw(20) <<"Inertial Properties "<<"||";
    std::cout<< std::right << std::setw(20) <<"Mass"<<"||"; // mass >0
    std::cout<< std::right << std::setw(12) <<"m cx";
    std::cout<< std::right << std::setw(12) <<"m cy";
    std::cout<< std::right << std::setw(12) <<"m cz"<<"||";
    std::cout<< std::right << std::setw(12) <<"Ixx";
    std::cout<< std::right << std::setw(12) <<"Ixy";
    std::cout<< std::right << std::setw(12) <<"Iyy";
    std::cout<< std::right << std::setw(12) <<"Ixz";
    std::cout<< std::right << std::setw(12) <<"Iyz";
    std::cout<< std::right << std::setw(12) <<"Izz"<<"||";
    std::cout<<std::endl;


    std::cout<<std::right << std::setw(20)<<"True Values "<<"||";
    std::cout<<std::right << std::setw(20)<<DynamicParameter_true(0)<<"||";
    std::cout<<std::right << std::setw(3)<<DynamicParameter_true.segment(1,3).transpose()<<"||";
    std::cout<<std::right << std::setw(3)<<DynamicParameter_true.segment(4,6).transpose()<<"||";
    std::cout<<std::endl;

    std::cout<<std::right << std::setw(20)<<"Estimated Values "<<"||";
    std::cout<<std::right << std::setw(20)<<DynamicParameter(0)<<"||";
    std::cout<<std::right << std::setw(3)<< DynamicParameter.segment(1,3).transpose()<<"||";
    std::cout<<std::right << std::setw(3)<< DynamicParameter.segment(4,6).transpose()<<"||";
    std::cout<<std::endl;

    std::cout<< checkPhysicalConsistency(getPseudoInertia(DynamicParameter_true)) <<std::endl;
    std::cout<< checkPhysicalConsistency(getPseudoInertia(DynamicParameter)) <<std::endl;

    std::cout << "-------------------------------------" <<std::endl;
  }
} // reduced model version
*/



void loadData(Eigen::MatrixXd& gc_matrix, Eigen::MatrixXd& gv_matrix, Eigen::MatrixXd& ga_matrix, Eigen::MatrixXd& tau_matrix){
  std::string rootPath = "/home/kdyun/Desktop/ros2_ws/src/analyzer/";
  std::string dataPath = "src/swing torque, free fall/";

  std::ifstream gc_log;
  gc_matrix.setZero(100000, 19);
  gc_log.open(rootPath + dataPath + "generalized_position.txt");
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t j = 0; j < 19; ++j) {
      gc_log >> gc_matrix(i, j);
    }
  }
  gc_log.close();
  std::cout << "gc log loaded" << std::endl;
  std::ifstream gv_log;
  gv_matrix.setZero(100000, 18);
  gv_log.open(rootPath + dataPath + "generalized_velocity.txt");
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t j = 0; j < 18; ++j) {
      gv_log >> gv_matrix(i, j);
    }
  }
  gv_log.close();
  std::cout << "gv log loaded" << std::endl;
  std::ifstream ga_log;
  ga_matrix.setZero(100000, 18);
  ga_log.open(rootPath + dataPath + "generalized_acceleration.txt");
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t j = 0; j < 18; ++j) {
      ga_log >> ga_matrix(i, j);
    }
  }
  ga_log.close();
  std::cout << "ga log loaded" << std::endl;
  std::ifstream tau_log;
  tau_matrix.setZero(100000, 18);
  tau_log.open(rootPath + dataPath + "generalized_torque.txt");
  for (size_t i = 0; i < 100000; ++i) {
    for (size_t j = 0; j < 18; ++j) {
      tau_log >> tau_matrix(i, j);
    }
  }
  tau_log.close();
  std::cout << "tau log loaded" << std::endl;
}


Eigen::VectorXd RaisimGcToPinocchioGc(Eigen::VectorXd raisim_gc){
  Eigen::VectorXd pin_gc;pin_gc.setZero(19);
  pin_gc.head(3) = raisim_gc.head(3);
  pin_gc.segment(3,4) << raisim_gc(4),raisim_gc(5),raisim_gc(6),raisim_gc(3); // (w,x,y,z) -> (x,y,z,w)

// (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_gc.tail(12).segment(0,3)=raisim_gc.tail(12).segment(3,3);
  pin_gc.tail(12).segment(3,3)=raisim_gc.tail(12).segment(0,3);
  pin_gc.tail(12).segment(6,3)=raisim_gc.tail(12).segment(9,3);
  pin_gc.tail(12).segment(9,3)=raisim_gc.tail(12).segment(6,3);
  return pin_gc;
}
Eigen::VectorXd RaisimGvToPinocchioGv(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_gv){
  Eigen::VectorXd pin_gv;pin_gv.setZero(18);
  raisim::Mat<3,3> rot;raisim::quatToRotMat(raisim_gc.segment(3,4),rot);
  Eigen::Matrix3d Rot; Rot.setIdentity();Rot = rot.e();

  pin_gv.head(3) = Rot.transpose()*raisim_gv.head(3); // world frame -> base local frame
  pin_gv.segment(3,3) = Rot.transpose()*raisim_gv.segment(3,3); // world frame -> base local frame
  // (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_gv.tail(12).segment(0,3)=raisim_gv.tail(12).segment(3,3);
  pin_gv.tail(12).segment(3,3)=raisim_gv.tail(12).segment(0,3);
  pin_gv.tail(12).segment(6,3)=raisim_gv.tail(12).segment(9,3);
  pin_gv.tail(12).segment(9,3)=raisim_gv.tail(12).segment(6,3);
  return pin_gv;
}
Eigen::VectorXd RaisimGaToPinocchioGa(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_gv,Eigen::VectorXd raisim_ga){
  Eigen::VectorXd pin_ga;pin_ga.setZero(18);
  raisim::Mat<3,3> rot;raisim::quatToRotMat(raisim_gc.segment(3,4),rot);
  Eigen::Matrix3d Rot,dRot;
  Rot.setIdentity(); dRot.setIdentity();
  Rot = rot.e(); dRot = skewMat(raisim_gv.segment(3,3))*Rot;

  pin_ga.head(3) = Rot.transpose()*raisim_ga.head(3) + dRot.transpose()*raisim_gv.head(3); // world frame -> base local frame
  pin_ga.segment(3,3) = Rot.transpose()*raisim_ga.segment(3,3)+ dRot.transpose()*raisim_gv.segment(3,3); // world frame -> base local frame
  // (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_ga.tail(12).segment(0,3)=raisim_ga.tail(12).segment(3,3);
  pin_ga.tail(12).segment(3,3)=raisim_ga.tail(12).segment(0,3);
  pin_ga.tail(12).segment(6,3)=raisim_ga.tail(12).segment(9,3);
  pin_ga.tail(12).segment(9,3)=raisim_ga.tail(12).segment(6,3);
  return pin_ga;
}
Eigen::VectorXd RaisimTauToPinocchioTau(Eigen::VectorXd raisim_gc,Eigen::VectorXd raisim_tau){
  Eigen::VectorXd pin_tau;pin_tau.setZero(18);
  raisim::Mat<3,3> rot;raisim::quatToRotMat(raisim_gc.segment(3,4),rot);
  Eigen::Matrix3d Rot; Rot.setIdentity();Rot = rot.e();

  pin_tau.head(3) = Rot.transpose()*raisim_tau.head(3); // world frame -> base local frame
  pin_tau.segment(3,3) = Rot.transpose()*raisim_tau.segment(3,3); // world frame -> base local frame
  // (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_tau.tail(12).segment(0,3)=raisim_tau.tail(12).segment(3,3);
  pin_tau.tail(12).segment(3,3)=raisim_tau.tail(12).segment(0,3);
  pin_tau.tail(12).segment(6,3)=raisim_tau.tail(12).segment(9,3);
  pin_tau.tail(12).segment(9,3)=raisim_tau.tail(12).segment(6,3);
  return pin_tau;
}
Eigen::MatrixXd RaisimMatToPinocchioMat(Eigen::VectorXd raisim_gc,Eigen::MatrixXd raisim_mat){
  Eigen::MatrixXd pin_mat1;pin_mat1.setZero(18,18);
  Eigen::MatrixXd pin_mat2;pin_mat2.setZero(18,18);

  raisim::Mat<3,3> rot;raisim::quatToRotMat(raisim_gc.segment(3,4),rot);
  Eigen::Matrix3d Rot; Rot.setIdentity();Rot = rot.e();
  // input transformation
  pin_mat1.leftCols(3) = raisim_mat.leftCols(3)*Rot;
  pin_mat1.middleCols(3,3) = raisim_mat.middleCols(3,3)*Rot;
  // (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_mat1.rightCols(12).middleCols(0,3) = raisim_mat.rightCols(12).middleCols(3,3);
  pin_mat1.rightCols(12).middleCols(3,3) = raisim_mat.rightCols(12).middleCols(0,3);
  pin_mat1.rightCols(12).middleCols(6,3) = raisim_mat.rightCols(12).middleCols(9,3);
  pin_mat1.rightCols(12).middleCols(9,3) = raisim_mat.rightCols(12).middleCols(6,3);

  // output transformation
  pin_mat2.topRows(3) = Rot.transpose()*pin_mat1.topRows(3);
  pin_mat2.middleRows(3,3) = Rot.transpose()*pin_mat1.middleRows(3,3);
  // (FR,FL,RR,RL) -> (FL,FR,RL,RR)
  pin_mat2.bottomRows(12).middleRows(0,3) = pin_mat1.bottomRows(12).middleRows(3,3);
  pin_mat2.bottomRows(12).middleRows(3,3) = pin_mat1.bottomRows(12).middleRows(0,3);
  pin_mat2.bottomRows(12).middleRows(6,3) = pin_mat1.bottomRows(12).middleRows(9,3);
  pin_mat2.bottomRows(12).middleRows(9,3) = pin_mat1.bottomRows(12).middleRows(6,3);

  return pin_mat2;
}

Eigen::VectorXd PinocchioGvToRaisimGv(Eigen::VectorXd pin_gc,Eigen::VectorXd pin_gv){
  Eigen::VectorXd raisim_gv;raisim_gv.setZero(18);

  Eigen::VectorXd quat_wxyz;quat_wxyz.setZero(4);
  quat_wxyz << pin_gc(6),pin_gc(3),pin_gc(4),pin_gc(5); // (x,y,z,w) -> (w,x,y,z)
  raisim::Mat<3,3> rot;raisim::quatToRotMat(quat_wxyz,rot);
  Eigen::Matrix3d Rot; Rot.setIdentity();Rot = rot.e();

  raisim_gv.head(3) = Rot*pin_gv.head(3); // base local frame -> world frame
  raisim_gv.segment(3,3) = Rot*pin_gv.segment(3,3); // base local frame -> world frame
  // (FL,FR,RL,RR) -> (FR,FL,RR,RL)
  raisim_gv.tail(12).segment(0,3)=pin_gv.tail(12).segment(3,3);
  raisim_gv.tail(12).segment(3,3)=pin_gv.tail(12).segment(0,3);
  raisim_gv.tail(12).segment(6,3)=pin_gv.tail(12).segment(9,3);
  raisim_gv.tail(12).segment(9,3)=pin_gv.tail(12).segment(6,3);
  return raisim_gv;
}

Eigen::Matrix3d skewMat(Eigen::Vector3d vec){
  Eigen::Matrix3d mat;
  mat <<
      0,-vec(2),vec(1),
      vec(2),0,-vec(0),
      -vec(1),vec(0),0;
  return mat;
}


Eigen::VectorXd getBarycentricParams(pinocchio::Model model){
  Eigen::VectorXd params(130);
  for (pinocchio::Model::JointIndex i = 1; i < (pinocchio::Model::JointIndex)model.njoints; ++i)
  {
  params.segment<10>((int)((i - 1) * 10)) = model.inertias[i].toDynamicParameters();
  }
  return params;
}

Eigen::VectorXd vech(Eigen::Matrix3d inertia){
  Eigen::VectorXd vec; vec.setZero(6);
  // Ixx, Ixy, Iyy, Ixz, Iyz, Izz
  vec << inertia(0,0),inertia(0,1),inertia(1,1),inertia(0,2),inertia(1,2),inertia(2,2);
  return vec;
}
Eigen::Matrix3d invvech(Eigen::VectorXd vec){
  Eigen::Matrix3d inertia;
  inertia <<
  vec(0),vec(1),vec(3),
  vec(1),vec(2),vec(4),
  vec(3),vec(4),vec(5);
  return inertia;
}
Eigen::Matrix4d getPseudoInertia(Eigen::VectorXd vec){
  //[m, mr_x, mr_y, mr_z, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz} ]
  Eigen::Matrix4d pseudo_inertia;
  Eigen::Matrix3d sigma;
  sigma = 0.5*(vec(4)+vec(6)+vec(9))*Eigen::Matrix3d::Identity()-invvech(vec.tail(6));
  pseudo_inertia.block(0,0,3,3)=sigma;
  pseudo_inertia.block(3,0,1,3)=vec.segment(1,3).transpose();
  pseudo_inertia.block(0,3,3,1)=vec.segment(1,3);
  pseudo_inertia.block(3,3,1,1)=vec.head(1);
  return pseudo_inertia;
}

bool checkPhysicalConsistency(Eigen::Matrix4d pseudoInertia){
//  std::cout << "pseudo inertia"<< std::endl << pseudoInertia << std::endl;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(pseudoInertia);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The eigenvalues are:\n" << eigensolver.eigenvalues().transpose() << std::endl;
//  std::cout << "Here's a matrix whose columns are eigenvectors  \n"
//            << "corresponding to these eigenvalues:\n"
//            << eigensolver.eigenvectors() << std::endl;
  return (eigensolver.eigenvalues().array()>0).all();
}


template <typename T>
void print_indent(std::string var_name, T var_value,               int indent)
{
  std::cout << std::setprecision(3) << std::left << std::setw(indent) << var_name << std::setw(8) << var_value << std::endl;
}
void print_indent(std::string var_name, Eigen::VectorXd var_value, int indent)
{
  int vec_size = var_value.size();
  std::stringstream ss;

  for(int i=0; i<vec_size-1; i++)
  {
    ss << std::setprecision(3) << std::left << std::setw(10) << var_value[i]; // ss << var_value[i] << "  ";
  }
  ss << std::setprecision(3) << std::left << var_value[vec_size-1];

  std::cout << std::left << std::setw(indent) << var_name << std::setw(indent) << ss.str() << std::endl;
}