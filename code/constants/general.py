# Environment type
GRIDWORLD = 0
ALE = 1
BOX2D = 2
MINATAR = 3
MAPE = 4
CLASSIC = 5
DW = 6  # 118 和 300节点罗伟的环境
SW = 7 # 电科院的123节点的环境

# Observation type
NONSPATIAL = 0
SPATIAL = 1

# Time limit
GRIDWORLD_TIMELIMIT = 100
ALE_TIMELIMIT = 27000
CLASSIC_TIMELIMIT = 1000
BOX2D_TIMELIMIT = 1000
MINATAR_TIMELIMIT = 1000
MAPE_TIMELIMIT = 100

# ENV_INFO
# Key:
# -- abbreviation
# -- env type (unused)
# -- obs type (unused)
# -- states are countable (unused)
# -- name
# -- difficulty_ramping (unused)
# -- level (unused)
# -- initial_difficulty (unused)
# -- maximum timesteps per episode

ENV_INFO = {
    'ALE-Adventure': ('ALE01V0', ALE, SPATIAL, False, 'AdventureNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-AirRaid': ('ALE02V0', ALE, SPATIAL, False, 'AirRaidNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Alien': ('ALE03V0', ALE, SPATIAL, False, 'AlienNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Amidar': ('ALE04V0', ALE, SPATIAL, False, 'AmidarNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Assault': ('ALE05V0', ALE, SPATIAL, False, 'AssaultNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Asterix': ('ALE06V0', ALE, SPATIAL, False, 'AsterixNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Asteroids': ('ALE07V0', ALE, SPATIAL, False, 'AsteroidsNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Atlantis': ('ALE08V0', ALE, SPATIAL, False, 'AtlantisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BankHeist': ('ALE09V0', ALE, SPATIAL, False, 'BankHeistNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BattleZone': ('ALE10V0', ALE, SPATIAL, False, 'BattleZoneNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-BeamRider': ('ALE11V0', ALE, SPATIAL, False, 'BeamRiderNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Berzerk': ('ALE12V0', ALE, SPATIAL, False, 'BerzerkNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Bowling': ('ALE13V0', ALE, SPATIAL, False, 'BowlingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Boxing': ('ALE14V0', ALE, SPATIAL, False, 'BoxingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Breakout': ('ALE15V0', ALE, SPATIAL, False, 'BreakoutNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Carnival': ('ALE16V0', ALE, SPATIAL, False, 'CarnivalNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Centipede': ('ALE17V0', ALE, SPATIAL, False, 'CentipedeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-ChopperCommand': ('ALE18V0', ALE, SPATIAL, False, 'ChopperCommandNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-CrazyClimber': ('ALE19V0', ALE, SPATIAL, False, 'CrazyClimberNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Defender': ('ALE20V0', ALE, SPATIAL, False, 'DefenderNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-DemonAttack': ('ALE21V0', ALE, SPATIAL, False, 'DemonAttackNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-DoubleDunk': ('ALE22V0', ALE, SPATIAL, False, 'DoubleDunkNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-ElevatorAction': ('ALE23V0', ALE, SPATIAL, False, 'ElevatorActionNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Enduro': ('ALE24V0', ALE, SPATIAL, False, 'EnduroNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-FishingDerby': ('ALE25V0', ALE, SPATIAL, False, 'FishingDerbyNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Freeway': ('ALE26V0', ALE, SPATIAL, False, 'FreewayNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Frostbite': ('ALE27V0', ALE, SPATIAL, False, 'FrostbiteNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Gopher': ('ALE28V0', ALE, SPATIAL, False, 'GopherNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Gravitar': ('ALE29V0', ALE, SPATIAL, False, 'GravitarNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Hero': ('ALE30V0', ALE, SPATIAL, False, 'HeroNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-IceHockey': ('ALE31V0', ALE, SPATIAL, False, 'IceHockeyNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Jamesbond': ('ALE32V0', ALE, SPATIAL, False, 'JamesbondNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-JourneyEscape': ('ALE33V0', ALE, SPATIAL, False, 'JourneyEscapeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Kaboom': ('ALE34V0', ALE, SPATIAL, False, 'KaboomNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Kangaroo': ('ALE35V0', ALE, SPATIAL, False, 'KangarooNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Krull': ('ALE36V0', ALE, SPATIAL, False, 'KrullNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-KungFuMaster': ('ALE37V0', ALE, SPATIAL, False, 'KungFuMasterNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-MontezumaRevenge': ('ALE38V0', ALE, SPATIAL, False, 'MontezumaRevengeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-MsPacman': ('ALE39V0', ALE, SPATIAL, False, 'MsPacmanNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-NameThisGame': ('ALE40V0', ALE, SPATIAL, False, 'NameThisGameNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Phoenix': ('ALE41V0', ALE, SPATIAL, False, 'PhoenixNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pitfall': ('ALE42V0', ALE, SPATIAL, False, 'PitfallNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pong': ('ALE43V0', ALE, SPATIAL, False, 'PongNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Pooyan': ('ALE44V0', ALE, SPATIAL, False, 'PooyanNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-PrivateEye': ('ALE45V0', ALE, SPATIAL, False, 'PrivateEyeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Qbert': ('ALE46V0', ALE, SPATIAL, False, 'QbertNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Riverraid': ('ALE47V0', ALE, SPATIAL, False, 'RiverraidNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-RoadRunner': ('ALE48V0', ALE, SPATIAL, False, 'RoadRunnerNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Robotank': ('ALE49V0', ALE, SPATIAL, False, 'RobotankNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Seaquest': ('ALE50V0', ALE, SPATIAL, False, 'SeaquestNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Skiing': ('ALE51V0', ALE, SPATIAL, False, 'SkiingNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Solaris': ('ALE52V0', ALE, SPATIAL, False, 'SolarisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-SpaceInvaders': ('ALE53V0', ALE, SPATIAL, False, 'SpaceInvadersNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-StarGunner': ('ALE54V0', ALE, SPATIAL, False, 'StarGunnerNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Tennis': ('ALE55V0', ALE, SPATIAL, False, 'TennisNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-TimePilot': ('ALE56V0', ALE, SPATIAL, False, 'TimePilotNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Tutankham': ('ALE57V0', ALE, SPATIAL, False, 'TutankhamNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-UpNDown': ('ALE58V0', ALE, SPATIAL, False, 'UpNDownNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Venture': ('ALE59V0', ALE, SPATIAL, False, 'VentureNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-VideoPinball': ('ALE60V0', ALE, SPATIAL, False, 'VideoPinballNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-WizardOfWor': ('ALE61V0', ALE, SPATIAL, False, 'WizardOfWorNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-YarsRevenge': ('ALE62V0', ALE, SPATIAL, False, 'YarsRevengeNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),
    'ALE-Zaxxon': ('ALE63V0', ALE, SPATIAL, False, 'ZaxxonNoFrameskip-v0', False, 0, 0, ALE_TIMELIMIT),

    'BOX2D-LunarLander': ('BOX2D00', BOX2D, NONSPATIAL, True, 'LunarLander-v2', False, 0, 0, BOX2D_TIMELIMIT),

    'GRIDWORLD-ReachTheGoal-v0': ('GW00V0', GRIDWORLD, NONSPATIAL, True, 'ReachTheGoal', False, 0, 0, 50),
    'GRIDWORLD-ReachTheGoal-v1': ('GW00V1', GRIDWORLD, NONSPATIAL, True, 'ReachTheGoal', False, 0, 0, 50),
    'GRIDWORLD-ReachTheGoal-v2': ('GW00V2', GRIDWORLD, SPATIAL, True, 'ReachTheGoal', False, 0, 0, 50),

    'MAPE-Simple': ('MAPE000', MAPE, NONSPATIAL, True, 'simple', False, 0, 0, MAPE_TIMELIMIT),

    'MinAtar-Asterix': ('MA00R', MINATAR, SPATIAL, False, 'asterix', False, 0, 0, MINATAR_TIMELIMIT),
    'MinAtar-Breakout': ('MA10R', MINATAR, SPATIAL, False, 'breakout', False, 0, 0, MINATAR_TIMELIMIT),
    'MinAtar-Freeway': ('MA20R', MINATAR, SPATIAL, False, 'freeway', False, 0, 0, MINATAR_TIMELIMIT),
    'MinAtar-Seaquest': ('MA30R', MINATAR, SPATIAL, False, 'seaquest', False, 0, 0, MINATAR_TIMELIMIT),
    'MinAtar-SpaceInvaders': ('MA40R', MINATAR, SPATIAL, False, 'space_invaders', False, 0, 0, MINATAR_TIMELIMIT),

    'DW-v1': ('DW00V1', DW, NONSPATIAL, True, 'S4case118', False, 0, 0, 50),

    'DW-v2': ('DW00V2', DW, NONSPATIAL, True, 'S4case300', False, 0, 0, 50),

    'DW-123': ('DW00V123', SW, NONSPATIAL, True, '', False, 0, 0, 50),
}


# Teacher Models (to load previously saved models as teachers):
# <Game Name>: (<Model directory>, <Model subdirectory (seed)>, <Checkpoint (timesteps)>), Network Structure (Type)

# Example: ALE24V0_EG_000_20201105-130625/0/model-6000000.ckpt will be loaded from "checkpoints" folder when the game
# is Enduro

TEACHER = {

    # Incompetent
    'ALE-Enduro-0': ('ALE24V0_EG_000_20201105-130625', '0', 1200e3, 0),
    'ALE-Freeway-0': ('ALE26V0_EG_000_20201105-172634', '0', 1800e3, 0),
    'ALE-Pong-0': ('ALE43V0_EG_000_20201106-011948', '0', 1800e3, 0),
    'ALE-Qbert-0': ('ALE46V0_EG_000_20201023-120616', '0', 1600e3, 0),
    'ALE-Seaquest-0': ('ALE50V0_EG_000_20201019-132350', '0', 2000e3, 0),

    # Competent
    'ALE-Enduro-1': ('ALE24V0_EG_000_20201105-130625', '0', 6000e3, 0),
    'ALE-Freeway-1': ('ALE26V0_EG_000_20201105-172634', '0', 3000e3, 0),
    'ALE-Pong-1': ('ALE43V0_EG_000_20201106-011948', '0', 5800e3, 0),
    'ALE-Qbert-1': ('ALE46V0_EG_000_20201023-120616', '0', 7000e3, 0),
    'ALE-Seaquest-1': ('ALE50V0_EG_000_20201019-132350', '0', 7000e3, 0),
    'ALE-Breakout-1': ('ALE15V0_000_660_20221011-071853', '0', 9800e3, 1),
    'ALE-SpaceInvaders-1':('ALE53V0_000_733_20230114-091421', '0', 7600e3, 1),

    # 'BOX2D-LunarLander': ('BOX2D00_000_174_20210527-180912', '100', 2000e3, 0),
    'BOX2D-LunarLander': ('BOX2D00_000_941_20210718-001159', '103', 400e3, 1),

    'DW-v1-1': ('DW00V1_000_783_20230524-073438', '0', 200000, 1),
    'DW-v2-1': ('DW00V2_000_351_20230618-152949', '0', 4e5, 1)
    
}

DEMONSTRATIONS_DATASET = {
    'ALE-Freeway': ('ALE26V0_EG_201_729_20210202-140852', '10'),
}
# Precomputed for 5000 observations
RND_MEAN_COEFFS = {
    'DW-v1': [0.22950383,],
    'DW-v2': [0.22950383,],
    'GridWorld': [0.01234568, 0.01234568, 0.17283951],
    'ALE-Enduro': [0.22950383, 0.2295122 , 0.2295207 , 0.22952849],
    'ALE-Freeway': [0.48930284, 0.48930272, 0.4893029 , 0.4893033],
    'ALE-Pong': [0.41700354, 0.41700402, 0.41700476, 0.41700545],
    'ALE-Qbert': [0.12600362, 0.12603626, 0.12606803, 0.12606308],
    'ALE-Seaquest': [0.18946025, 0.18946102, 0.18946168, 0.18946236],
}

# Precomputed for 5000 observations
RND_STD_COEFFS = {
    'DW-v1': [0.18300994],
    'DW-v2': [0.22950383,],
    'GridWorld': [0.11042311, 0.11042311, 0.37810846],
    'ALE-Enduro': [0.18300994, 0.18301411, 0.18301918, 0.18302345],
    'ALE-Freeway': [0.19500406, 0.19500417, 0.19500428, 0.1950046 ],
    'ALE-Pong': [0.18529113, 0.18529119, 0.18529128, 0.18529138],
    'ALE-Qbert': [0.19381808, 0.19378734, 0.19375697, 0.19374232],
    'ALE-Seaquest': [0.1649532 , 0.16495302, 0.16495296, 0.16495283],
}