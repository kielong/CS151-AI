# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print(successorGameState)
        # input(type(successorGameState))
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Always go to the closest food
        newFood = successorGameState.getFood().asList()
        # Each food is a location tuple
        minFoodist = 9999.
        for food in newFood:
            minFoodist = min(minFoodist, manhattanDistance(newPos, food))

        # Don't step onto the square if there is a ghost
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -9999.
        # reciprocal
        return successorGameState.getScore() + 1.0/minFoodist

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.maxstep(gameState, 0, 0)[0]

    # Since we can't use the word "depth", I would like to call it "layer"
    def minimax(self, gameState, agentIndex, layer):
        # Return gameState if either search or game is over
        if (layer is self.depth * gameState.getNumAgents()) or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxstep(gameState, agentIndex, layer)[1]
        else:
            return self.minstep(gameState, agentIndex, layer)[1]

    def maxstep(self, gameState, agentIndex, layer):
        # First element is the direction; second is the score
        bestAction = ("Begin", -9999.)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                               (layer+1) % gameState.getNumAgents(), layer+1))
            if succAction[1] > bestAction[1]:
                bestAction = succAction
        return bestAction

    def minstep(self, gameState, agentIndex, layer):
        # bestAction's direction will be replaced by East, West, South and North
        bestAction = ("Begin", 9999.)
        for action in gameState.getLegalActions(agentIndex):
            succAction = (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                               (layer + 1) % gameState.getNumAgents(), layer+1))
            if succAction[1] < bestAction[1]:
                bestAction = succAction
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        return self.maxstep(gameState, agentIndex=0, layer=0, alpha=-9999., beta=9999.)[0]

    def alphabeta(self, gameState, agentIndex, layer, alpha, beta):
        if layer is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.maxstep(gameState, agentIndex, layer, alpha, beta)[1]
        else:
            return self.minstep(gameState, agentIndex, layer, alpha, beta)[1]

    def maxstep(self, gameState, agentIndex, layer, alpha, beta):
        # Direction and the alpha value. Alpha is initialized as a very negative number
        bestAction = ("Begin", -9999.)
        for action in gameState.getLegalActions(agentIndex):
            nextAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (layer+1) % gameState.getNumAgents(), layer+1, alpha, beta))
            bestAction = max(bestAction, nextAction, key=lambda x: x[1])
            # Prunning
            if bestAction[1] > beta:
                # If this happens, no need to look further. Just stop here by placing this value in
                return bestAction
            else:
                # renew alpha if bestAction is more optimal
                alpha = max(alpha, bestAction[1])
        return bestAction

    def minstep(self, gameState, agentIndex, layer, alpha, beta):
        bestAction = ("min", float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            nextAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (layer+1) % gameState.getNumAgents(), layer+1, alpha, beta))
            bestAction = min(bestAction, nextAction, key=lambda x: x[1])
            # Prunning, similar to maxstep case
            if bestAction[1] < alpha:
                return bestAction
            else:
                beta = min(beta, bestAction[1])
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(gameState, agent, depth):
            # Return gameState if either search or game is over
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0:  # maximizing for pacman
                bestAction = -9999.
                for action in gameState.getLegalActions(agent):
                    if expectimax(gameState.generateSuccessor(agent, action), 1, depth) > bestAction:
                        bestAction = expectimax(gameState.generateSuccessor(agent, action), 1, depth)
                return bestAction
            else:  # this else statement takes care of the expectimax (chance nodes) action of the ghosts
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                    depth += 1
                temp_sum = 0
                for action in gameState.getLegalActions(agent):
                    temp_sum += expectimax(gameState.generateSuccessor(agent, action), nextAgent, depth )
                return temp_sum/len(gameState.getLegalActions(agent))
                

        # The following code does the maximizing task for pacman
        maximum = -9999.
        # The bestAction can be initialized to any action
        bestAction = Directions.STOP
        for agentState in gameState.getLegalActions(0):
            score = expectimax(gameState.generateSuccessor(0, agentState), 1, 0)
            if score > maximum or maximum == -9999.:
                maximum = score
                bestAction = agentState

        return bestAction
    


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Essentially did the same thing as the original evaluation function,
    except this "better" evaluation function looks at the current game state, rather 
    than the successor states. Therefore, this "better" function evaluates states, 
    rather than actions. We start by initializing the new positions. Next, we calculated
    the distance to the closest food pellet. After, we calculate the distances from
    pacman to the ghosts, checking if its distance is 1 around pacman. Finally, we just
    return a combination of these metrics.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Always go to the closest food
    newFood = currentGameState.getFood().asList()
    # Each food is a location tuple
    minFoodist = 9999.
    for food in newFood:
        minFoodist = min(minFoodist, manhattanDistance(newPos, food))

    # Don't step onto the square if there is a ghost
    for ghost in currentGameState.getGhostPositions():
        if (manhattanDistance(newPos, ghost) < 1):
            return -9999.
    # reciprocal
    return currentGameState.getScore() + 1.0/minFoodist

# Abbreviation
better = betterEvaluationFunction