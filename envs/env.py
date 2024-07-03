class Env:

    """
    This class defines all the keyword arguments in the custom wrappers and eval_env flag
    """

    def __init__(
        self,
        timeout_steps: int = -1,
        fail_on_timeout: bool = False,
        eval_env: bool = False,
    ):
        # time_wrapper kwargs
        self.timeout_steps = timeout_steps
        self.fail_on_timeout = fail_on_timeout
        # eval_env
        self.eval_env = eval_env
