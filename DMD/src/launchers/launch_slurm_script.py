import subprocess
import time
import argparse

def submit_job(script_path):
    """Submit a SLURM job and return the job ID."""
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode == 0:
        # Extract job ID from output
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        raise Exception("Failed to submit job: " + result.stderr)

def check_job_status(job_id):
    """Check the status of a SLURM job."""
    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader'], capture_output=True, text=True)
    if result.returncode == 0:
        # Extract job state from output
        status = result.stdout.strip()
        return status
    else:
        raise Exception("Failed to check job status: " + result.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, default='', help='path to your SLURM script')
    args = parser.parse_args()

    script_path = args.script
    job_id = submit_job(script_path)
    print(f"Submitted job {job_id} (save this on Notion to keep track of this file.)")
    print('Also, save Python process ID to keep track of this process.')

    while True:
        status = check_job_status(job_id)
        # print(f"Current status of job {job_id}: {status}")

        if 'COMPLETED' in status:
            print("Job completed successfully.")
            break        
        elif 'FAILED' in status or 'CANCELLED' in status or 'TIMEOUT' in status:
            print(f"Job {job_id} ended with status {status}. Resubmitting...")
            job_id = submit_job(script_path)
            print(f"Resubmitted job {job_id}")
        else:
            time.sleep(1800)  # Wait for 30 minutes (1800 seconds) before checking again
            # time.sleep(5) # for debugging... wait for 5 seconds before checking again

if __name__ == "__main__":
    main()