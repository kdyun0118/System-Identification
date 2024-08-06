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
  // v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities+
  // a = [local_base_acceleration_linear, local_base_acceleration_angular, joint_accelerations]
  pinocchio::Model Go1_pin_model;
  pinocchio::urdf::buildModel(urdf_filename,pinocchio::JointModelFreeFlyer(),Go1_pin_model);

  Go1_pin_model.gravity.linear(gravity_vector);

  pinocchio::Data Go1_pin_data = pinocchio::Data(Go1_pin_model);

  /// load Data
  Eigen::MatrixXd gc_matrix,gv_matrix,ga_matrix,tau_matrix;
  loadData(gc_matrix,gv_matrix,ga_matrix,tau_matrix);

  for (int i = 0; i < 1000; ++i) {
    std::cout << i << " -----------------------------------------" << std::endl;
    /// raisim dynamics test
    Eigen::VectorXd raisim_gc, raisim_gv, raisim_ga, raisim_tau;
    raisim_gc.setZero(19);
    raisim_gv.setZero(18);
    raisim_ga.setZero(18);
    raisim_tau.setZero(18);
    raisim_gc = gc_matrix.row(i);
    raisim_gv = gv_matrix.row(i);
    raisim_ga = ga_matrix.row(i);
    raisim_tau = tau_matrix.row(i);

    raisim::Mat<3, 3> rot;
    raisim::quatToRotMat(raisim_gc.segment(3, 4), rot);
//  std::cout << "rotation matrix by raisim" << std::endl << rot.e() << std::endl;
//  std::cout << "raisim gc " <<  raisim_gc.transpose()<< std::endl;
//  std::cout << "raisim gv " <<  raisim_gv.transpose() << std::endl;
//  std::cout << "raisim ga " <<  raisim_ga.transpose() << std::endl;
//  std::cout << "raisim tau "<<  raisim_tau.transpose() << std::endl;

    Go1_raisim->setState(raisim_gc, raisim_gv);
    Eigen::MatrixXd M = Go1_raisim->getMassMatrix().e();
    Eigen::VectorXd h = Go1_raisim->getNonlinearities(gravity_vector).e();
    Eigen::MatrixXd tmp = RaisimMatToPinocchioMat(raisim_gc, M);
    Eigen::VectorXd tmp2 = RaisimTauToPinocchioTau(raisim_gc, h);

//  std::cout << "raisim M to pin" << std::endl << tmp << std::endl;
//  std::cout << "raisim h to pin" << std::endl <<  tmp2.transpose() << std::endl;
//  std::cout << "raisim error " <<  (M*raisim_ga+h-raisim_tau).transpose()<< std::endl;
    std::cout << "  raisim error norm " << (M * raisim_ga + h - raisim_tau).norm() << std::endl;
//    std::cout << std::endl;

//    std::cout << i << " -----------------------------------------" << std::endl;
    /// pinocchio dynamics test
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
//  std::cout << "pin gc " <<  pin_gc.transpose()<< std::endl;
//  std::cout << "pin gv " <<  pin_gv.transpose() << std::endl;
//  std::cout << "pin ga " <<  pin_ga.transpose() << std::endl;
//  std::cout << "pin tau "<<  pin_tau.transpose() << std::endl;

    pinocchio::crba(Go1_pin_model, Go1_pin_data, pin_gc, pinocchio::Convention::LOCAL); // fill only upper triangular
//    pinocchio::crba(Go1_pin_model,Go1_pin_data,pin_gc,pinocchio::Convention::WORLD); // fill only upper triangular
    Go1_pin_data.M.triangularView<Eigen::StrictlyLower>() =
        Go1_pin_data.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv); // run this after rnea?

//  std::cout << "crba_local " <<std::endl<<  Go1_pin_data.M << std::endl;
//  std::cout << "nle " <<std::endl<<  Go1_pin_data.nle.transpose() << std::endl;

//  std::cout << "pin error " <<  (Go1_pin_data.M*pin_ga+Go1_pin_data.nle-pin_tau).transpose() << std::endl;
    std::cout << "  pin error norm " << (Go1_pin_data.M * pin_ga + Go1_pin_data.nle - pin_tau).norm() << std::endl;

//  std::cout << "M diff " <<std::endl<<  (Go1_pin_data.M-tmp).norm() << std::endl;
//  std::cout << "h diff " <<std::endl<<  (Go1_pin_data.nle-tmp2).transpose() << std::endl;

//  /// aba test
//  pinocchio::aba(Go1_pin_model, Go1_pin_data, pin_gc, pin_gv, pin_tau);
//  std::cout << "aba " <<  Go1_pin_data.ddq.transpose() << std::endl;
//  std::cout << "aba error1 " <<  (Go1_pin_data.ddq-pin_ga).transpose() << std::endl;
//  std::cout << "aba error2 " <<  (Go1_pin_data.M*(Go1_pin_data.ddq-pin_ga)).transpose() << std::endl;
  }
}

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
