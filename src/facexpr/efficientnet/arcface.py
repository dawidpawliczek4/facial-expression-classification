# TODO: arcface
# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features, s=30.0, m=0.50):
#         super().__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)
#         self.s, self.m = s, m

#     def forward(self, x, labels=None):
#         cosine = F.linear(F.normalize(x), F.normalize(self.weight))
#         if labels is None:
#             return cosine * self.s
#         theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
#         target_logits = torch.cos(theta + self.m)
#         one_hot = torch.zeros_like(cosine)
#         one_hot.scatter_(1, labels.view(-1,1), 1.0)
#         output = self.s * (one_hot * target_logits + (1.0 - one_hot) * cosine)
#         return output
