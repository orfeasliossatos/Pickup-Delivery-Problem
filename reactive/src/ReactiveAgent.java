import java.util.*;
import java.util.stream.Collectors;

import logist.simulation.Vehicle;
import logist.agent.Agent;
import logist.behavior.ReactiveBehavior;
import logist.plan.Action;
import logist.plan.Action.Move;
import logist.plan.Action.Pickup;
import logist.task.Task;
import logist.task.TaskDistribution;
import logist.topology.Topology;
import logist.topology.Topology.City;

public class ReactiveAgent implements ReactiveBehavior {

	private Random random;
	private double pPickup;
	private int numActions = 0;
	private Agent agent;
	private int costPerKm;
	private List<RLAState> states = new ArrayList<RLAState>();
	private List<RLAAction> actions = new ArrayList<RLAAction>();
	private Map<Pair<RLAState, RLAAction>, Double> rewards = new HashMap<>();
	private Map<Triple<RLAState, RLAAction, RLAState>, Double> transitions = new HashMap<>();
	private Map<RLAState, Double> prevValues = new HashMap<>();
	private Map<RLAState, Double> currValues = new HashMap<>();
	private Map<RLAState, RLAAction> bestActions = new HashMap<>();
	@Override
	public void setup(Topology topology, TaskDistribution td, Agent agent) {
		// Called exactly once before the simulation starts and before
		// any other method is called.

		// Create the possible states
		for (City from : topology.cities()) {
			for (City to : topology.cities()) {
				if (!from.equals(to))
					states.add(new RLAState(from, to));
			}
			states.add(new RLAState(from, null));
		}

		// Create the possible actions
		actions.add(new RLAAction(true, null));
		for (City dest : topology.cities()) {
			actions.add(new RLAAction(false, dest));
		}

		// Check agent vehicles.
		this.agent = agent;
		this.costPerKm = agent.vehicles().get(0).costPerKm();

        /* Fill out the rewards mapping which returns a non-infinite value for defined state-action pairs. */
		for (RLAState state : states) {
			for (RLAAction action : actions) {

				City from = state.from;
				City to = state.to;
				City dest = action.dest;

				// 1. If you take then the task must exist.
				// 2. If you don't take the task then you must move to a neighbour.
				if ((action.take && to == null) || (!action.take && !from.hasNeighbor(dest)))
					continue;

				// Taking the package gives a reward. Taking or not still incurs travel cost.
				double reward = (action.take && to != null) ? td.reward(from, to) - from.distanceTo(to) * costPerKm
						: - from.distanceTo(dest) * costPerKm;

				// Fill in reward table
				rewards.put(new Pair<>(state, action), reward);
			}
		}

		/*
		 Fill out the state transition mapping which returns a value in [0, 1]
		 for well-defined curr-action-next triples. Undefined triples map to 0.
		*/
		for (RLAState curr : states) {
			for (RLAAction action : actions) {


				// 1. If you take then the task must exist.
				// 2. If you don't take the task then you must move to a neighbour.
				if ((action.take && curr.to == null) || (!action.take && !curr.from.hasNeighbor(action.dest)))
					continue;

				for (RLAState next : states) {

					// If you take then you get the next potential task at the task's finish.
					if ((action.take && !curr.to.equals(next.from))
							|| (!action.take && !action.dest.equals(next.from)))
						continue;

					// Each "next" state represents a Task(from, to). But if the current task was dropped
					// then the next "from" city will be the alternative destination.
					double probability =  td.probability(action.take ? next.from : action.dest, next.to);

					// Fill in reward table
					transitions.put(new Triple<>(curr, action, next), probability);
				}
			}
		}

		// Transition table correctness : The sum over "next" states should be 1 for every action.
		boolean correctness = transitions.entrySet().stream()
				.collect(Collectors.groupingBy(
						entry -> entry.getKey().dropThird(), // Group by current state + action
						Collectors.summingDouble(Map.Entry::getValue) // Sum the values
				)).values().stream().allMatch(value -> (Math.abs(value - 1) < 0.0001)); // Check if sums to 1
		assert(correctness);
		
		// Reads the discount factor from the agents.xml file.
		// If the property is not present it defaults to 0.95
		Double discount = agent.readProperty("discount-factor", Double.class,
				0.95);


		// Value iteration
		int i = 0;
		while (true) {
			for (RLAState curr : states) {

				RLAAction bestAction = null;
				Double bestQValue = Double.NEGATIVE_INFINITY;

				for (RLAAction action : actions) {
					double qValue = rewards.getOrDefault(new Pair<>(curr, action), Double.NEGATIVE_INFINITY)
							+  discount * states.stream().mapToDouble(next -> currValues.getOrDefault(next, 0.0)
							* transitions.getOrDefault(new Triple<>(curr, action, next), 0.0)).sum();

					// Save intermediate best action and q-value
					if (qValue > bestQValue) {
						bestAction = action;
						bestQValue = qValue;
					}
				}

				// Save values and best actions
				currValues.put(curr, bestQValue);
				bestActions.put(curr, bestAction);
			}

			// Check for convergence
			Double valueDiff = 0.0;
			for (RLAState state : currValues.keySet()) {
				Double prevValue = prevValues.getOrDefault(state,0.0);
				Double currValue = currValues.getOrDefault(state, 0.0);
				valueDiff += Math.abs(currValue - prevValue);
				prevValues.put(state, currValue);
			}

			// At least 100 iterations. Continue until convergence.
			if (i > 100 && valueDiff == 0) {
				System.out.println("Done.");
				break;
			}

			if (i % 10 == 0)
				System.out.println("Iteration " + String.valueOf(i) + ". Value difference: " + String.valueOf(valueDiff));

			i++;
		}

		// Print where the vehicle likes to go in case of no task.
		System.out.println("When no task, the prefered destinations are");
		StringBuilder builder = new StringBuilder("[");
		for (RLAState state : bestActions.keySet()) {
			if (state.to == null) {
				builder.append(state.from.toString() + " -> " + bestActions.get(state).dest.toString() + ", ");
			}
		}
		builder.append("]");
		System.out.println(builder);

		// Print any towns that are preferably skipped (particularly bad tasks)
		System.out.println("These tasks are always skipped because taking gives low rewards: ");
		builder = new StringBuilder("[");
		for (RLAState state : bestActions.keySet()) {
			if (state.to != null && !bestActions.get(state).take) {
				builder.append(state.toString() + ": $" + rewards.get(new Pair<>(state, new RLAAction(true, null))).toString() + ", ");
			}
		}
		builder.append("]");
		System.out.println(builder);


		this.random = new Random();
	}

	@Override
	public Action act(Vehicle vehicle, Task availableTask) {

		// Convert to state representation.
		RLAState curr = (availableTask == null) ? new RLAState(vehicle.getCurrentCity(), null)
				: new RLAState(availableTask.pickupCity, availableTask.deliveryCity);

		// Look up and perform best action pre-computed.
		RLAAction bestAction = bestActions.get(curr);
		Action action = (bestAction.take) ? new Pickup(availableTask) : new Move(bestAction.dest);

		// action = (bestAction.take) ? new Pickup(availableTask) : new Move(vehicle.getCurrentCity().randomNeighbor(random));

		// Print current state
		if (numActions >= 1) {
			System.out.println("The total profit after "+numActions+" actions is "+agent.getTotalProfit()+" (average profit: "+(agent.getTotalProfit() / (double)numActions)+")");
		}
		numActions++;
		
		return action;
	}


	// State class for the reinforcement learning algorithm
	private static class RLAState {
		private City from;
		private City to;

		public RLAState(City from, City to) {
			this.from = from;
			this.to = to;
		}

		@Override public String toString() {
			return String.format("State[%s, %s]", from, to);
		}

		@Override public boolean equals(Object obj) {

			if (obj == null) {
				return false;
			}

			if (obj.getClass() != this.getClass()) {
				return false;
			}

			final RLAState other = (RLAState) obj;

			if ((this.to == null) != (other.to == null)) {
				return false;
			}

			return this.from.equals(other.from) && ((this.to == null) ? true : this.to.equals(other.to));
		}

		@Override public int hashCode() {
			int hash = 7;
			hash = 31 * hash + ((to == null) ? 0 : to.hashCode());
			hash = 31 * hash + ((from == null) ? 0 : from.hashCode());
			return hash;
		}
	}


	// Action class for the reinforcement learning algorithm
	private static class RLAAction {
		private Boolean take;
		private City dest;
		public RLAAction(Boolean take, City dest) {
			this.take = take;
			this.dest = dest;
		}
		@Override public String toString() {
			if (this.take) {
				return new String("Action[take]");
			} else {
				return String.format("Action[%s]", dest);
			}
		}
		@Override public boolean equals(Object obj) {

			if (obj == null) {
				return false;
			}

			if (obj.getClass() != this.getClass()) {
				return false;
			}

			final RLAAction other = (RLAAction) obj;

			if ((this.dest == null) != (other.dest == null)) {
				return false;
			}

			return this.take.equals(other.take) && ((this.dest==null) ? true : this.dest.equals(other.dest));
		}

		@Override public int hashCode() {
			int hash = 7;
			hash = 31 * hash + ((take == null) ? 0 : take.hashCode());
			hash = 31 * hash + ((dest == null) ? 0 : dest.hashCode());
			return hash;
		}
	}

	private static class Pair<T, U> {
		T first;
		U second;
		public Pair(T first, U second) {
			this.first = first;
			this.second = second;
		}
		@Override public String toString() {
			return String.format("(%s, %s)", first, second);
		}
		@Override public boolean equals(Object obj) {

			if (obj == null) {
				return false;
			}

			if (obj.getClass() != this.getClass()) {
				return false;
			}

			final Pair<T, U> other = (Pair<T, U>) obj;

			return (this.first.equals(other.first) && this.second.equals(other.second));
		}

		@Override public int hashCode() {
			int hash = 7;
			hash = 31 * hash + ((first == null) ? 0 : first.hashCode());
			hash = 31 * hash + ((second == null) ? 0 : second.hashCode());
			return hash;
		}
	}

	private static class Triple<T, U, V> {
		T first;
		U second;
		V third;
		public Triple(T first, U second, V third) {
			this.first = first;
			this.second = second;
			this.third = third;
		}
		public Pair<T, U> dropThird() {
			return new Pair<T, U>(first, second);
		}
		@Override public String toString() {
			return String.format("(%s, %s, %s)", first, second, third);
		}
		@Override public boolean equals(Object obj) {

			if (obj == null) {
				return false;
			}

			if (obj.getClass() != this.getClass()) {
				return false;
			}

			final Triple<T, U, V> other = (Triple<T, U, V>) obj;

			return (this.first.equals(other.first)
					&& this.second.equals(other.second)
					&& this.third.equals(other.third));
		}
		@Override public int hashCode() {
			int hash = 7;
			hash = 31 * hash + ((first == null) ? 0 : first.hashCode());
			hash = 31 * hash + ((second == null) ? 0 : second.hashCode());
			hash = 31 * hash + ((third == null) ? 0 : third.hashCode());
			return hash;
		}
	}
}
