function y = sat(u,lim)

y = min([max([u, -lim]), lim]);