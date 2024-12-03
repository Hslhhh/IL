from pymycobot import MyArmM

# 示例
myarmm = MyArmM('/dev/ttyACM0')


# 获取所有关节的当前角度
angles = myarmm.get_joints_angle()
print(f"当前所有的关节角度是: {angles}")

angles0 = myarmm.get_robot_modified_version()
print(angles0)