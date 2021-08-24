import subprocess
import time


def run_commands(commands_list, max_concurrent_jobs=4):
    """
    Runs a list of shell jobs concurrently using the subprocess.Popen interface
    Arguments:
        commands_list: a list where each element is a list of args to subprocess.Popen
        max_concurrent_jobs: maximum number of jobs to run at once
    """
    max_iterations = len(commands_list)
    if max_concurrent_jobs > max_iterations:
        max_concurrent_jobs = max_iterations
    i = 0
    current_jobs_running = 0
    completed_jobs = 0

    job_dict = {}
    running_job_ids = []

    while completed_jobs < max_iterations:
        # Start jobs until reaching maximum number of concurrent jobs
        while (current_jobs_running < max_concurrent_jobs) and (i < max_iterations):
            print("Starting job {}".format(i))
            job_dict[i] = subprocess.Popen(commands_list[i])
            running_job_ids.append(i)
            current_jobs_running += 1
            i += 1

        # Check if jobs are done
        time.sleep(5)
        still_running_job_ids = []
        for j in running_job_ids:
            if job_dict[j].poll() is None:
                still_running_job_ids.append(j)
            else:
                job_dict[j].wait()
                print("Job {} complete".format(j))
                del job_dict[j]
                current_jobs_running -= 1
                completed_jobs += 1
        running_job_ids = still_running_job_ids


def flatten_multicolumns(df):
    """
    Converts multi-index columns into single column
    """
    df.columns = [
        "_".join([el for el in col if el != ""]).strip()
        for col in df.columns.values
        if len(col) > 1
    ]
    return df
