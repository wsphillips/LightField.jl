using LightField

lf = setup("configexample.toml")
phasespacepsf = propagate(lf)

printf("Run complete")
