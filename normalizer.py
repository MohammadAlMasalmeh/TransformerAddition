def normalize(user_input):
    user_nums = user_input.split('+')
    user_nums = [x for x in user_nums if x.strip() != ""]
    val1 = int(user_nums[0].strip())
    val2 = int(user_nums[1].strip().strip('=').strip())
    return val1, val2