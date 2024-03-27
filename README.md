# Maize Seed Quality Classification with DeepSMOTE and EfficientNet

#### List of Content
<ul>
  <li>Method of Research</li>
  <li>Dataset Source</li>
  <li>Summary</li>
</ul>

## Method of Research :
<ol>
  <li> I start from Input <b>Dataset</b> </li>
  <li> Oversampling Dataset with DeepSMOTE</li>
  <li> Splitting Dataset into Training Set, Validation Set, and Testing Set</li>
  <li> Then, Im trying to Train Model with Training Set and Validation Set with EfficientNet and k-Fold Cross Validation </li>
  <li> Finally im testing EfficientNet Model</li>
</ol>

## Dataset Source :
<ol>
  <li> Thats Journal of Dataset <a href="https://arxiv.org/pdf/2110.00777">(source)</a></li>
  <li> Dataset <a href="https://naagar.github.io/cornseedsdataset/">(source)</a> </li>
</ol>

## Summary :
<table>
  <tr>
    <th>Experiments</th>
    <th>with DeepSMOTE</th>
    <th>without DeepSMOTE</th>
  </tr>
  <tr>
    <td>EfficientNet B0</td>
    <td>79.20% </td>
    <td>67.28%</td>
  </tr>
  <tr>
    <td>EfficientNet B1</td>
    <td>69.35%</td>
    <td>65.57%</td>
  </tr>
  <tr>
    <td>EfficientNet B2</td>
    <td>70.04%</td>
    <td>663.57%</td>
  </tr>
</table>
