import pandas as pd
import numpy as np
from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
from statistical_fns import t_statistic, p_value
from comet_ml import Experiment
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv('api_key')
PROJECT_NAME=os.getenv('project_name')
WORKSPACE=os.getenv('workspace')


class VarianceReductionDemo(FlowSpec):
    """
    Variance Reduction Demo takes real data, manipulates a treatment
    and then demonstrates how variance reduction can increase sensitivity
    """

    CONTROL_FILE = Parameter(
        name='control_file',
        help='Name of control dataframe pickle file',
        default='control_df.pkl')

    TREATMENT_FILE = Parameter(
        name='treatment_file',
        help='Name of treatment dataframe pickle file',
        default='treatment_df.pkl')

    treatment_bump = Parameter(
        name='treatment_bump',
        help='Fraction of samples that will be bumped to 1.  '
             'Most are zero, so this is effectively an increase fraction',
        default=0.0015)

    rho_start = Parameter(
        name='rho_start',
        help='Correlation of the covariate lower bound.',
        default=0.00)

    rho_end = Parameter(
        name='rho_end',
        help='Correlation of the covariate upper bound.',
        default=0.99)

    rho_step = Parameter(
        name='rho_step',
        help='Correlation of the covariate step.',
        default=0.05)

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        self.rhos = np.arange(self.rho_start, self.rho_end, self.rho_step)

        self.next(self.de_pickle)

    @step
    def de_pickle(self):
        """
        Unpack pickle files
        """
        self.df_control = pd.read_pickle(self.CONTROL_FILE)
        self.df_treatment = pd.read_pickle(self.TREATMENT_FILE)

        self.next(self.add_covariate, foreach='rhos')

    @step
    def add_covariate(self):
        """
        Add covariate with correlation rho.
        """
        self.rho = self.input
        df = pd.concat([self.df_control, self.df_treatment], axis=0)

        # creating correlated covariate
        z = (df['target'] - df['target'].mean()) / df['target'].std()
        rnd = np.random.normal(size=len(df))
        df['x'] = self.rho * z + np.sqrt(1 - self.rho ** 2) * rnd

        print(df[['x','target']].corr()['x'][1])
        self.df_control = df.loc[self.df_control.index].copy()
        self.df_treatment = df.loc[self.df_treatment.index].copy()

        self.next(self.modify_treatment)

    @step
    def modify_treatment(self):
        bln_samples = self.df_treatment.sample(frac=self.treatment_bump).index
        self.df_treatment.loc[bln_samples, 'target'] = 1

        self.next(self.calculate_stats)


    @step
    def calculate_stats(self):
        N = len(self.df_treatment)
        self.ratio_no_reduction = t_statistic(self.df_control, self.df_treatment, var_type='total')
        self.ratio_cuped = t_statistic(self.df_control, self.df_treatment, var_type='cuped')
        self.p_value_no_reduction = p_value(self.ratio_no_reduction, N)
        self.p_value_cuped = p_value(self.ratio_cuped, N)

        self.next(self.log_stats)


    @step
    def log_stats(self):
        params = {"CUPED_correlation": self.rho,
              "treatment_bump_fraction": self.treatment_bump
              }
        metrics = {
            "No_Var_Reduction_t":self.ratio_no_reduction,
            "CUPED_Reduction_t": self.ratio_cuped,
            "No_Var_Reduction_p": self.p_value_no_reduction,
            "CUPED_Reduction_p": self.p_value_cuped
        }

        exp = Experiment(
            api_key=API_KEY,
            project_name=PROJECT_NAME,
            workspace=WORKSPACE)
        exp.log_parameters(params)
        exp.log_metrics(metrics)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)


    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    VarianceReductionDemo()
