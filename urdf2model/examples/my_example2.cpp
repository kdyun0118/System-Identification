#include <casadi/casadi.hpp>
#include "model_interface.hpp"

int main()
{
    // Example with Kinova Gen3 URDF.

    // ---------------------------------------------------------------------
    // Create a model based on a URDF file
    // ---------------------------------------------------------------------
      std::string urdf_filename = "../urdf2model/models/Hound_new_heavy/Hound_model.urdf";
    // Instantiate a Serial_Robot object called robot_model
      mecali::Serial_Robot robot_model;
    // Create the model based on a URDF file
      robot_model.import_model(urdf_filename);

    // ---------------------------------------------------------------------
    // Look inside the robot_model object. What variables can you fetch?
    // ---------------------------------------------------------------------
    // Get some variables contained in the robot_model object
      std::string name      = robot_model.name;
      int         n_q       = robot_model.n_q;
      int         n_joints  = robot_model.n_joints;
      int         n_dof     = robot_model.n_dof;
      int         n_frames  = robot_model.n_frames;
      std::vector<std::string> joint_names = robot_model.joint_names;
      std::vector<std::string> joint_types = robot_model.joint_types;
      Eigen::VectorXd          gravity     = robot_model.gravity;
      Eigen::VectorXd          joint_torque_limit    = robot_model.joint_torque_limit;
      Eigen::VectorXd          joint_pos_ub          = robot_model.joint_pos_ub;
      Eigen::VectorXd          joint_pos_lb          = robot_model.joint_pos_lb;
      Eigen::VectorXd          joint_vel_limit       = robot_model.joint_vel_limit;
      Eigen::VectorXd          neutral_configuration = robot_model.neutral_configuration;
    // Print some information related to the imported model (boundaries, frames, DoF, etc)
      robot_model.print_model_data();

    // ---------------------------------------------------------------------
    // Set functions for robot dynamics and kinematics
    // ---------------------------------------------------------------------
    // Set function for forward dynamics
      casadi::Function fwd_dynamics = robot_model.forward_dynamics();
    // Set function for inverse dynamics
      casadi::Function inv_dynamics = robot_model.inverse_dynamics();

    // Set functions for mass_inverse matrix, coriolis matrix, and generalized gravity vector
      casadi::Function gen_gravity = robot_model.generalized_gravity();

      casadi::Function coriolis = robot_model.coriolis_matrix();

      casadi::Function mass_inverse = robot_model.mass_inverse_matrix();

    // Set function for joint torque regressor: regressor(q, dq, ddq)*barycentric_params = tau
      casadi::Function regressor = robot_model.joint_torque_regressor();

    // Set function for forward kinematics
      // The forward kinematics function can be set in multiple ways
      // Calling forward_kinematics without any argument generates a function which outputs a transformation matrix for each frame in the robot.
      casadi::Function fk_T_1 = robot_model.forward_kinematics();
      // The first optional argument refers to the content of the output function: it can be set to be "position", "rotation", or "transformation"
      // Setting the first argument as "transformation" is just the same as not including any argument: outputs a 4x4 T matrix for each frame.
      casadi::Function fk_T_2 = robot_model.forward_kinematics("transformation");
      // Setting the first argument as "position" means that the function is going to output a 3x1 position vector for each frame.
      casadi::Function fk_pos = robot_model.forward_kinematics("position");
      // Setting the first argument as "rotation" means that the function is going to output a 3x3 rotation matrix for each frame.
      casadi::Function fk_rot = robot_model.forward_kinematics("rotation");

      // You can also generate a F.K. function for specific frames (using the frame name or index, which you can see after executing "robot_model.print_model_data()"" )
      casadi::Function fk_T_multiframes_by_name  = robot_model.forward_kinematics("transformation", std::vector<std::string>{"RR_foot", "RR_knee_joint", "RR_roll_joint"});
      casadi::Function fk_T_multiframes_by_index = robot_model.forward_kinematics("transformation", std::vector<int>{34, 31, 27});
      casadi::Function fk_pos_oneframe_by_name   = robot_model.forward_kinematics("position", "RR_foot");
      casadi::Function fk_pos_oneframe_by_index  = robot_model.forward_kinematics("position", 18);

    // ---------------------------------------------------------------------
    // Generate random configuration vectors
    // ---------------------------------------------------------------------
      // You can generate a random configuration vector (size n_q) that takes into account upper and lower boundaries of the joint angles.
      Eigen::VectorXd random_conf_vector = robot_model.randomConfiguration();
      // You can also use your own upper and lower boundaries of the joint angles using Eigen::VectorXd of size n_dof,
      Eigen::VectorXd random_conf_vector_bounded = robot_model.randomConfiguration(-0.94159*Eigen::VectorXd::Ones(robot_model.n_dof), 0.94159*Eigen::VectorXd::Ones(robot_model.n_dof));
      // or using std::vector<double> of size n_dof.
      Eigen::VectorXd random_conf_vector_bounded2 = robot_model.randomConfiguration(std::vector<double>{-2, -2.2, -3.0, -2.4, -2.5, -2.6, -1, -2.2, -3.0, -2.4, -2.5, -2.6}, std::vector<double>{2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 0.75, 2.1, 2.2, 2.3, 2.4, 2.5});

      // mecali::print_indent("Random configuration (bounded) = ", random_conf_vector_bounded, 38);

    // ---------------------------------------------------------------------
    // Evaluate a kinematics or dynamics function
    // ---------------------------------------------------------------------
    // Test a function with numerical values
      /* Create a std::vector<double> of size robot_model.n_q (take care in case there are continuous joints in your model)

         For the Kinova Gen3 robot n_dof = 7, but n_q = 11, since {q1. q3. q5. q7} are continuous (unbounded) joints.
         Continuous joints are not represented just by q_i, but by [cos(q_i), sin(q_i)].
         The configuration vector is then set as: [cos(q1), sin(q1), q2, cos(q3), sin(q3), q4, cos(q5), sin(q5), q6, cos(q7), sin(q7)]
      */
//      std::vector<double> q_vec = {0.86602540378, 0.5, 0, 1, 0, -0.45, 1, 0, 0.2, 1, 0};
      std::vector<double> q_vec = {0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0,};
      // Evaluate the function with a casadi::DMVector containing q_vec as input
      casadi::DM pos_res = fk_pos_oneframe_by_name(casadi::DMVector {q_vec})[0];
      std::cout << "Function result with q_vec input        : " << pos_res << std::endl;

      // You can also use robot's neutral configuration as input
      std::vector<double> q_vec_neutral((size_t)robot_model.n_q);
      Eigen::Map<mecali::ConfigVector>( q_vec_neutral.data(), robot_model.n_q, 1 ) = robot_model.neutral_configuration; // Populate q_vec_neutral with the robot's neutral configuration

      casadi::DM pos_neutral = fk_pos_oneframe_by_name(casadi::DMVector {q_vec_neutral})[0];
      std::cout << "Function result with q_vec_neutral input: " << pos_neutral << std::endl;

      // or use a random configuration as input
      std::vector<double> q_vec_random((size_t)robot_model.n_q);
      Eigen::Map<mecali::ConfigVector>( q_vec_random.data(), robot_model.n_q, 1 ) = robot_model.randomConfiguration(); // Populate q_vec_neutral with a random configuration

      casadi::DM pos_random  = fk_pos_oneframe_by_name(casadi::DMVector {q_vec_random})[0];
      std::cout << "Function result with q_vec_random input : " << pos_random  << std::endl;

    // ---------------------------------------------------------------------
    // Generate (or save) a function
    // ---------------------------------------------------------------------
    // Code-generate or save a function
      // If not setting options, function fk_T_1 (or any function) will only be C-code-generated as "first_function.c" (or any other name you set)
      // mecali::generate_code(fk_T_1, "first_function");
      // If you use options, you can set if you want to C-code-generate the function, or just save it as "second_function.casadi" (which can be loaded afterwards using casadi::Function::load("second_function.casadi"))
      // mecali::Dictionary codegen_options;
      // codegen_options["c"]=false;
      // codegen_options["save"]=true;
      // mecali::generate_code(fk_T_multiframes_by_name, "second_function", codegen_options);

    // ------------------------------------------------------------------------
    // Create a reduced model based on a URDF file and list of joints by name
    // ------------------------------------------------------------------------

    // Instantiate a Serial_Robot object called robot_model
      mecali::Serial_Robot reduced_robot_model_1;
    // Define list of joints to be locked (by name)
      std::vector<std::string> list_of_joints_to_lock_by_name = {"RR_roll_joint","RR_hip_joint","RR_knee_joint","RR_foot_fixed"};
    // Define (optinal) robot configuration where joints should be locked
      Eigen::VectorXd q_init_1 = robot_model.neutral_configuration;
    // Define (optinal) gravity vector to be used
      Eigen::Vector3d gravity_vector_1(0,0,-9.81);
    // Create the model based on a URDF file
      // reduced_robot_model_1.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_name);
      // reduced_robot_model_1.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_name, q_init_1);
      reduced_robot_model_1.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_name, q_init_1, gravity_vector_1);

    // ------------------------------------------------------------------------
    // Create a reduced model based on a URDF file and list of joints by index
    // ------------------------------------------------------------------------
    // Instantiate a Serial_Robot object called robot_model
      mecali::Serial_Robot reduced_robot_model_2;
    // Define list of joints to be locked (by index)
      // std::vector<std::size_t> list_of_joints_to_lock_by_id = {1,3,5};
      std::vector<int> list_of_joints_to_lock_by_idint = {3,4,5,6,7};
    // Define (optinal) robot configuration where joints should be locked
//      std::vector<double> q_init_vec_2 = {1, 0, 0.6, 1, 0, -0.4, 1, 0, 0.3, 1, 0};
//      Eigen::VectorXd q_init_2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(q_init_vec_2.data(), q_init_vec_2.size());
      Eigen::VectorXd q_init_2 = robot_model.neutral_configuration;

  // Define (optinal) gravity vector to be used
      Eigen::Vector3d gravity_vector_2(0,0,-9.81);
    // Create the model based on a URDF file
      // reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_id);
      // reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_id, q_init_2,);
      // reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_id, q_init_2, gravity_vector_2);
      // reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_idint);
      // reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_idint, q_init_2);
      reduced_robot_model_2.import_reduced_model(urdf_filename, list_of_joints_to_lock_by_idint, q_init_2, gravity_vector_2);

      reduced_robot_model_2.print_model_data();

      std::cout << "home config for reduced: " << reduced_robot_model_2.neutral_configuration << std::endl;
      std::cout << "forward dynamics: " << reduced_robot_model_2.forward_dynamics() << std::endl;
      std::cout << "forward kinematics: " << reduced_robot_model_2.forward_kinematics("position", "RR_foot") << std::endl;
      std::cout << "barycentric parameters: " << reduced_robot_model_2.barycentric_params.transpose() << std::endl;

}
