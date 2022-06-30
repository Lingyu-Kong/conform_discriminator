
def plot_smoother(batch_steps, batch_energy):
    energies = []
    steps = []
    for i in range(len(batch_steps)):
        if(batch_energy[i] <= 5):
            energies.append(batch_energy[i])
            steps.append(batch_steps[i])
    return steps, energies