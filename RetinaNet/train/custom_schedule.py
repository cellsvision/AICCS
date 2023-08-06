import math
import copy

class LinearWarmUp():
    def __init__(self, schedule, start_lr, length, start_iter=0):
        """
        schedule: a pre-initialized schedule (e.g. TriangularSchedule(min_lr=0.5, max_lr=2, cycle_length=500))
        start_lr: learning rate used at start of the warm-up (float)
        length: number of iterations used for the warm-up (int)
        """
        self.schedule = schedule
        self.start_lr = start_lr
        # calling mx.lr_scheduler.LRScheduler effects state, so calling a copy
        self.finish_lr = copy.copy(schedule)(0)
        self.length = length
        self.start_iter = start_iter

    def __call__(self, iteration):
        if iteration + self.start_iter <= self.length:
            return (iteration+self.start_iter) * (self.finish_lr - self.start_lr)/(self.length) + self.start_lr
        else:
            return self.schedule(iteration + self.start_iter - self.length)


class CosineCyclicSchedule():
    def __init__(self, min_lr, max_lr, cycle_length, start_iter=0):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = [cycle_length] if isinstance(cycle_length, int) else cycle_length
        self.start_iter = start_iter

    def __call__(self, iteration):
        iteration = self.start_iter + iteration
        if len(self.cycle_length) == 1:
          current_cycle_index = 0
        else:
          current_cycle_index = len([x for x in self.cycle_length if x > iteration])
        remainder = iteration % (self.cycle_length[current_cycle_index] + 1)
        unit_cycle = (1 + math.cos(remainder * math.pi / self.cycle_length[current_cycle_index])) / 2
        adjusted_cycle = (unit_cycle * (self.max_lr - self.min_lr)) + self.min_lr
        return adjusted_cycle

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    #plt.scatter(iterations, lrs, linewidth=1.)
    plt.plot(iterations, lrs, linewidth=1.)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()
  


  epoch = 3600
  begin_epoch = 13
  end_epoch = 50 + 2
  cycle_epoch = 10
  
  cycle_length = cycle_epoch*epoch
  
  start_iter = begin_epoch * epoch
  end_iter = (end_epoch - begin_epoch) * epoch
  
  
  schedule = CosineCyclicSchedule(min_lr=.00001, max_lr=.01, start_iter=0, cycle_length=cycle_length)
  
  warmup_schedule = LinearWarmUp(schedule, 0, epoch*2, start_iter=start_iter)
  
  plot_schedule(warmup_schedule, iterations=end_iter)
  #plot_schedule(schedule, iterations=end_iter)