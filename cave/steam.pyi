
# Due to Steamworks SDK TOS, Cave can't publicly redistribute Steam APIs, so in order to use 
# Cave Engine's Steam Integration, make sure you first:
#
# - Have a valid Steamworks Account with a valid App and APPID (https://partner.steamgames.com/)
# - Have downloaded the Steamworks SDK: https://partner.steamgames.com/doc/sdk
# - Understand how the Steam API works: https://partner.steamgames.com/doc/sdk/api
#
# After that, you will need to extract the steam_api64.dll (Windows) or libsteam_api.so (Linux)
# into the same folder that Cave is located. Then Cave should be able to dinamically load the
# SDK and enable the APIs bellow for you. If you don't, they will silently fail to execute.
#
# Please read and refer to Steamworks SDK Documentation to better understand it. Cave's Integration
# was built on top ot Steamworks SDK v1.6.1 and it simplifies it a bit. If you miss certain
# functionality from the SDK, please contact us and explain which functionality you want and
# the use case for it. 
# 
# IMPORTANT (from Steam Docs):
# The Steamworks API will not initialize if it does not know the App ID of your game. 
# When you launch your app from Steam itself then it will automatically have the App ID available. 
# While developing you will need to hint this to Steam with a text file. Create the a text file called 
# "steam_appid.txt" next to your executable containing just the App ID and nothing else. 
# This overrides the value that Steam provides. You should not ship this with your builds.
#
# You can use the App ID "480" for testing purposes. Good to read:
# https://partner.steamgames.com/doc/features/achievements/ach_guide
#
# For Cave Engine, the "steam_appid.txt" should be located at the same directory as the engine, not
# your project directory. This is due to how Steam API works and can't be changed.

# Cave Engine's Steam integration, allows you to implement features such as Steam Achievements. Please check the python stub (steam.pyi) for more information on how to Setup the Steamworks SDK with Cave Engine in order to use this integration.

def restartAppIfNecessary(appID: int) -> bool: ... # Checks if your executable was launched through Steam and relaunches it through Steam if it wasn't. This is optional but highly recommended as the Steam context associated with your application (including your App ID) will not be set up if the user launches the executable directly. If you choose to use this then it should be the first Steamworks function call you make, right before cave.steam.init. If it returns true, you should consider quitting the game since Steam will already reopen it.
def init() -> int: ... # Initializes Steam and setup everything for you. You MUST call this before starting to use the other interfaces. You only need to call this once, unless you shut it down.
def shutdown() -> None: ... # When you are done using the Steamworks API you should call shutdown to release the resources used by your application internally within Steam. You should call this during process shutdown if possible, when you're about to close the game'.
def isInitialized() -> bool: ... # Checks if the Steam API was initialized (init) and the SDK  (dll/so) was loaded properly.
def getPersonaName() -> str: ... # Returns the current user's Steam name.
def unlockAchievement(achievementID: str) -> bool: ... 
def hasAchievement(achievementID: str) -> bool: ... 
def clearAchievement(achievementID: str) -> bool: ... 
def getNumAchievements() -> int: ... 
def getAchievementName(id: int) -> str: ... 
def isOverlayActive() -> bool: ... 

